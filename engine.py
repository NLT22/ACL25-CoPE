import torch
import torch.nn.functional as F
from util.misc import MetricLogger, SmoothedValue
import copy


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def compute_probabilistic_distances(query_features, target_means, target_vars):
    """
    Compute probabilistic distances between query and all targets using the 
    same distance metric as ClosedFormSampledDistanceLoss.
    
    Args:
        query_features: dict with 'mean' and 'var' keys [batch_size, dim]
        target_means: tensor [num_targets, dim] 
        target_vars: tensor [num_targets, dim] (log var values)
    
    Returns:
        distances: tensor [batch_size, num_targets] (lower = more similar)
    """
    # Expand dimensions for broadcasting
    query_mean = query_features['mean'].unsqueeze(1)  # [batch, 1, dim]
    query_var = query_features['var'].unsqueeze(1)    # [batch, 1, dim]
    target_means = target_means.unsqueeze(0)          # [1, num_targets, dim]
    target_vars = target_vars.unsqueeze(0)            # [1, num_targets, dim]
    
    # Mean distance (squared Euclidean between normalized features)
    # This matches the mu_pdist computation in ClosedFormSampledDistanceLoss
    mu_pdist = ((query_mean - target_means) ** 2).sum(-1)
    
    # Uncertainty distance (sum of standard deviations)
    # Note: inputs are log(var), so exp() converts to var
    # This matches the sigma_pdist computation in ClosedFormSampledDistanceLoss
    sigma_pdist = (torch.exp(query_var) + torch.exp(target_vars)).sum(-1)
    
    # Total probabilistic distance (lower = more similar)
    distances = mu_pdist + sigma_pdist
    
    return distances


def train_one_epoch(model, model_params, ema_params, data_loader, optimizer, device, epoch, config, logger=None):
    """
    Training function for probabilistic models. Supports warmup phase using cross-entropy loss.
    """
    model.train()
    optimizer.zero_grad()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # Check if we're in warmup phase
    warmup_epochs = getattr(config.training, 'warmup_epochs', 0)
    use_warmup = epoch < warmup_epochs and hasattr(model, 'warmup')
    header = 'Epoch: [{}]'.format(epoch)

    for batch in metric_logger.log_every(data_loader, 50, header):

        ref_imgs, texts, tgt_imgs = batch['ref_img'], batch['text_instruction'], batch['tgt_img']

        ref_imgs = ref_imgs.to(device)
        tgt_imgs = tgt_imgs.to(device)
        input_ids = model.tokenize(texts, padding='max_length', return_tensors='pt').input_ids.to(device)

        with torch.amp.autocast('cuda'):
            if use_warmup:
                loss = model.warmup(ref_imgs, input_ids, tgt_imgs)
            else:
                loss = model(ref_imgs, input_ids, tgt_imgs)

        # Check for NaN loss and raise exception with step information
        if torch.isnan(loss):
            current_step = metric_logger.meters['loss'].count + 1  # +1 since we haven't updated the meter yet
            training_mode = "warmup" if use_warmup else "probabilistic"
            print(f"NaN loss detected at epoch {epoch}, step {current_step} (mode: {training_mode})")
            print(f"Batch info: ref_imgs.shape={ref_imgs.shape}, tgt_imgs.shape={tgt_imgs.shape}")
            print(f"Loss value: {loss.item()}")
            
            # Check for NaN in model parameters
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
            if nan_params:
                print(f"NaN found in parameters: {nan_params}")
            
            # Check for NaN in gradients if available
            nan_grads = []
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    nan_grads.append(name)
            if nan_grads:
                print(f"NaN found in gradients: {nan_grads}")
                
            raise ValueError(f"NaN loss encountered at epoch {epoch}, step {current_step} (mode: {training_mode})")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_ema(ema_params, model_params, rate=config.ema.rate)

        metric_logger.update(loss=loss.item())

    # Log training-specific metrics
    if use_warmup:
        pass
    else:
        # Log probabilistic-specific metrics if available
        if hasattr(model, 'pcmepp_loss'):
            if logger:
                logger.info(f"Epoch: {epoch}, Loss: {loss.item()}, "
                           f"shift: {model.pcmepp_loss.shift.item():.3f}, "
                           f"negative_scale: {model.pcmepp_loss.negative_scale.item():.3f}")
        else:
            if logger:
                logger.info(f"Epoch: {epoch}, Loss: {loss.item()}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_probabilistic(model, ema_params, val_loader, tgt_loader, device, config, logger=None, use_ema=True, use_probabilistic=True):
    """
    Evaluation function for probabilistic models that return dictionary features.
    
    Args:
        use_probabilistic: If True, uses probabilistic distance. If False, uses cosine similarity of feat['mean'].
    """
    model.eval()
    metrics = config.validation.metrics

    recall_metrics = {}
    for m in metrics:
        recall_metrics[m] = 0

    # Apply EMA if requested
    if use_ema:
        model_state_dict = copy.deepcopy(model.state_dict())
        ema_state_dict = copy.deepcopy(model.state_dict())
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        model.load_state_dict(ema_state_dict)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            
            # Collect all target features - handle dictionary returns
            all_tgt_names = []
            all_tgt_features = []
            
            logger.info("Collecting target features...")
            for batch in tgt_loader:
                all_tgt_names += batch['img_name']
                tgt_feat = model.encode_target(batch['img'].to(device))
                all_tgt_features.append(tgt_feat)
            
            # Separate means and vars for efficient batch processing
            all_tgt_means = torch.cat([feat['mean'] for feat in all_tgt_features])
            all_tgt_vars = torch.cat([feat['var'] for feat in all_tgt_features])
            
            logger.info(f"Collected {len(all_tgt_names)} target features")
            
            count = 0
            
            # Process validation batches
            logger.info("Processing validation queries...")
            for batch in val_loader:
                ref_imgs, texts, tgt_img_names = batch['ref_img'], batch['text_instruction'], batch['tgt_img_name']

                ref_imgs = ref_imgs.to(device)
                input_ids = model.tokenize(texts, padding='max_length', return_tensors='pt').input_ids.to(device)

                # Get query features (returns dict with 'mean' and 'var')
                query_features = model.encode_query(ref_imgs, input_ids)
                
                # Find target indices 
                tgt_idx = torch.tensor([all_tgt_names.index(name) for name in tgt_img_names], 
                                     device=query_features['mean'].device)

                if use_probabilistic:
                    # Compute probabilistic distances
                    distances = compute_probabilistic_distances(query_features, all_tgt_means, all_tgt_vars)
                    # Sort by distance (ascending - lower distance = more similar)
                    sorted_indices = torch.argsort(distances, dim=-1)
                else:
                    # Use cosine similarity of mean features
                    query_mean = query_features['mean']  # [batch_size, dim]
                    # Normalize features for cosine similarity
                    query_mean_norm = F.normalize(query_mean, p=2, dim=1)
                    tgt_means_norm = F.normalize(all_tgt_means, p=2, dim=1)
                    # Compute cosine similarity [batch_size, num_targets]
                    similarities = torch.mm(query_mean_norm, tgt_means_norm.t())
                    # Sort by similarity (descending - higher similarity = more similar)
                    sorted_indices = torch.argsort(similarities, dim=-1, descending=True)

                # Create a mask of correct matches
                labels = (sorted_indices == tgt_idx.unsqueeze(1)).float()

                # Compute Recall@K
                for metric in metrics:
                    recall_metrics[metric] += torch.sum(labels[:, :metric]).item()

                count += len(ref_imgs)

    # Restore original model state
    if use_ema:
        model.load_state_dict(model_state_dict)

    # Compute final recall percentages
    for metric in metrics:
        recall_metrics[metric] = (recall_metrics[metric] / count) * 100

    return recall_metrics


