import torch
import logging
import argparse
import copy
import os
import json
import shutil
from pathlib import Path
from torch.optim import AdamW
from omegaconf import OmegaConf
from util.data import build_data
from engine import train_one_epoch, evaluate_probabilistic
from models import model_registry


def check_for_nan_parameters(model, logger=None, stage="model"):
    """
    Comprehensive check for NaN values in model parameters.
    
    Args:
        model: PyTorch model to check
        logger: Logger instance for reporting
        stage: String describing when this check is being performed
    
    Returns:
        bool: True if no NaN found, False otherwise
    
    Raises:
        ValueError: If NaN parameters are found
    """
    nan_params = []
    inf_params = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)
    
    if nan_params or inf_params:
        error_parts = []
        if nan_params:
            error_parts.append(f"NaN parameters: {nan_params}")
        if inf_params:
            error_parts.append(f"Inf parameters: {inf_params}")
        
        error_msg = f"Invalid parameters found in {stage} - {'; '.join(error_parts)}"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    if logger:
        logger.info(f"✓ {stage} parameter check passed - no NaN/Inf values found")
    
    return True


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file using OmegaConf"""
    config = OmegaConf.load(config_path)
    return config


def save_checkpoint_hf(model, epoch, eval_stats, config, logger, checkpoint_dir, is_best=False, is_last=False):
    """
    Save checkpoint in HuggingFace format.
    
    Args:
        model: PyTorch model to save
        epoch: Current epoch number
        eval_stats: Evaluation statistics
        config: Training configuration
        logger: Logger instance
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
        is_last: Whether this is the last epoch
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine checkpoint name
    if is_best:
        checkpoint_name = "best_model"
    elif is_last:
        checkpoint_name = "last_checkpoint"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}"
    
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict in HuggingFace format
    model_state_path = checkpoint_path / "pytorch_model.bin"
    torch.save(model.state_dict(), model_state_path)
    
    # Save model config (if available)
    if hasattr(model, 'config'):
        model_config_path = checkpoint_path / "config.json"
        with open(model_config_path, 'w') as f:
            json.dump(model.config, f, indent=2)
    
    # Save metadata
    metadata = {
        'epoch': epoch,
        'eval_stats': eval_stats,
        'is_best': is_best,
        'is_last': is_last,
        'model_name': config.model.name,
        'dataset': config.data.dataset
    }
    
    metadata_path = checkpoint_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def cleanup_old_checkpoints(checkpoint_dir, max_checkpoints, logger):
    """
    Remove old checkpoints to maintain max_checkpoints limit.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        logger: Logger instance
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    # Find all epoch checkpoints (exclude best_model and last_checkpoint)
    epoch_checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint_epoch_"):
            try:
                epoch_num = int(item.name.split("_")[-1])
                # Use modification time for sorting, not epoch number
                mod_time = item.stat().st_mtime
                epoch_checkpoints.append((mod_time, epoch_num, item))
            except ValueError:
                continue
    
    # Sort by modification time (oldest first) and remove oldest if exceeding limit
    epoch_checkpoints.sort(key=lambda x: x[0])
    
    # If max_checkpoints is 0, remove all epoch checkpoints
    if max_checkpoints == 0:
        checkpoints_to_remove = epoch_checkpoints.copy()
        epoch_checkpoints.clear()
    else:
        checkpoints_to_remove = []
        while len(epoch_checkpoints) > max_checkpoints:
            checkpoints_to_remove.append(epoch_checkpoints.pop(0))
    
    # Remove the identified checkpoints
    for mod_time, epoch_num, checkpoint_path in checkpoints_to_remove:
        try:
            # Remove the entire checkpoint directory
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed old checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")


def should_save_checkpoint(epoch, config, eval_stats, best_metric_value, logger):
    """
    Determine whether to save a checkpoint based on configuration.
    
    Args:
        epoch: Current epoch
        config: Training configuration
        eval_stats: Current evaluation statistics
        best_metric_value: Best metric value seen so far
        logger: Logger instance
    
    Returns:
        Tuple of (should_save, is_best, new_best_metric_value)
    """
    checkpoint_config = getattr(config.output, 'checkpoint', None) if hasattr(config, 'output') else None
    if not checkpoint_config:
        return False, False, best_metric_value
    
    save_frequency = getattr(checkpoint_config, 'save_frequency', 10)
    save_best_only = getattr(checkpoint_config, 'save_best_only', True)
    best_metric = getattr(checkpoint_config, 'best_metric', 'r_10')
    best_metric_mode = getattr(checkpoint_config, 'best_metric_mode', 'max')
    
    # Check if it's time to save based on frequency
    is_frequency_save = (epoch + 1) % save_frequency == 0
    
    # Check if current model is the best
    is_best = False
    new_best_metric_value = best_metric_value
    
    if best_metric in eval_stats:
        current_metric_value = eval_stats[best_metric]
        
        if best_metric_value is None:
            is_best = True
            new_best_metric_value = current_metric_value
        elif best_metric_mode == 'max' and current_metric_value > best_metric_value:
            is_best = True
            new_best_metric_value = current_metric_value
        elif best_metric_mode == 'min' and current_metric_value < best_metric_value:
            is_best = True
            new_best_metric_value = current_metric_value
    
    # Decide whether to save
    if save_best_only:
        should_save = is_best
    else:
        should_save = is_frequency_save or is_best
    
    if is_best:
        logger.info(f"New best model! {best_metric}: {new_best_metric_value:.4f} (previous: {best_metric_value})")
    
    return should_save, is_best, new_best_metric_value


def main(config):
    # misc.init_distributed_mode(config)
    device = torch.device(config.device)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)

    # TODO: Set random seeds for reproducibility

    # Model - ensure it's a probabilistic model
    # Pass model-specific parameters from config
    neighborhood_config = getattr(config.model, 'neighborhood_loss', {})
    
    # Loss weight configuration for combined training
    loss_weights_config = getattr(config.model, 'loss_weights', {})
    
    model_kwargs = {
        'neighborhood_loss_weight': getattr(neighborhood_config, 'weight', 0.01),
        'k_neighbors': getattr(neighborhood_config, 'k_neighbors', 5),
        'alpha': getattr(loss_weights_config, 'alpha', 0.9),
        'beta': getattr(loss_weights_config, 'beta', 0.1)
    }
    
    # Add other configurable parameters
    preprocessing_config = getattr(config.model, 'preprocessing', {})
    model_kwargs.update({
        'target_ratio': getattr(preprocessing_config, 'target_ratio', 1.25),
        'image_size': getattr(preprocessing_config, 'image_size', 224),
        'local_files_only': getattr(config.model, 'local_files_only', True)
    })
    
    model = model_registry.get_model(config.model.name, config.model.path, **model_kwargs).to(device)
    
    # Verify this is a probabilistic model
    if not hasattr(model.backbone, 'loss'):
        logger.warning(f"Model {config.model.name} does not have backbone.loss attribute. "
                      "This script is designed for probabilistic models like ProbCLIPModel.")
    
    # Apply CoPE loss configuration from config if specified
    if hasattr(model.backbone, 'loss') and hasattr(config.model, 'cope_loss'):
        cope_config = config.model.cope_loss
        logger.info("Applying CoPE loss configuration from config...")
        
        # Apply configuration parameters to the correct loss object
        if hasattr(cope_config, 'init_shift'):
            model.backbone.loss.shift.data.fill_(cope_config.init_shift)
            logger.info(f"Set CoPE shift to: {cope_config.init_shift}")
            
        if hasattr(cope_config, 'init_negative_scale'):
            model.backbone.loss.negative_scale.data.fill_(cope_config.init_negative_scale)
            logger.info(f"Set CoPE negative_scale to: {cope_config.init_negative_scale}")
    
    # Check for NaN parameters before training
    check_for_nan_parameters(model, logger, "initial model after config application")
    
    # Check that model returns dictionary features
    logger.info("Verifying model returns dictionary features...")
    model.eval()
    
    model_params = list(model.parameters())
    ema_params = copy.deepcopy(model_params)
    
    # Check EMA parameters
    logger.info("Checking EMA parameters for NaN values...")
    for i, (param, ema_param) in enumerate(zip(model_params, ema_params)):
        if torch.isnan(ema_param).any():
            raise ValueError(f"NaN found in EMA parameter {i} during initialization")
        if torch.isinf(ema_param).any():
            raise ValueError(f"Inf found in EMA parameter {i} during initialization")
    
    logger.info("✓ EMA parameters initialized successfully")

    # Build datasets and dataloaders
    data_components = build_data(config, preprocess=model.preprocess)
    train_loader = data_components['train_loader']
    val_loader = data_components['val_loader']
    target_loader = data_components['target_loader']

    # Log available models
    available_models = model_registry.list_models()
    logger.info(f"Available models: {available_models}")
    logger.info(f"Selected model: {config.model.name}")

    # Optimizer - Handle CoPE loss parameters specially
    if hasattr(model.backbone, 'loss') and model.backbone.loss is not None:
        # Separate CoPE loss parameters from other model parameters
        cope_params = list(model.backbone.loss.parameters())
        cope_param_ids = {id(p) for p in cope_params}
        other_params = [p for p in model.parameters() if id(p) not in cope_param_ids]
        
        # Use special learning rate for CoPE parameters if specified in config
        cope_lr = getattr(config.optimizer, 'cope_lr', config.optimizer.lr)
        
        param_groups = [
            {'params': other_params, 'lr': config.optimizer.lr,
             'betas': (config.optimizer.beta1, config.optimizer.beta2), 'eps': config.optimizer.eps},
            {'params': cope_params, 'lr': cope_lr,
             'betas': (config.optimizer.beta1, config.optimizer.beta2), 'eps': config.optimizer.eps,
             'weight_decay': 0.0}
        ]
        logger.info(f"Using CoPE loss with special learning rate: {cope_lr}")
        logger.info(f"CoPE parameters: {len(cope_params)} (shift, negative_scale)")
    else:
        param_groups = [{'params': model.parameters(), 'lr': config.optimizer.lr,
                        'betas': (config.optimizer.beta1, config.optimizer.beta2), 'eps': config.optimizer.eps}]
    
    optimizer = AdamW(param_groups)
    scaler = torch.amp.GradScaler('cuda')

    # Warmup configuration
    warmup_epochs = getattr(config.training, 'warmup_epochs', 0)
    if warmup_epochs > 0:
        if hasattr(model, 'warmup'):
            logger.info(f"Warmup training enabled for {warmup_epochs} epochs")
            logger.info("Will use cross-entropy loss during warmup phase")
        else:
            logger.warning(f"Warmup epochs specified ({warmup_epochs}) but model does not have warmup method")
            logger.warning("Proceeding with standard probabilistic training")
    
    # Initialize checkpoint tracking
    best_metric_value = None
    checkpoint_config = getattr(config.output, 'checkpoint', None) if hasattr(config, 'output') else None
    if checkpoint_config:
        logger.info(f"Checkpoint saving enabled - saving to: {checkpoint_config.save_dir}")
        logger.info(f"Save frequency: {checkpoint_config.save_frequency}, "
                   f"Best metric: {checkpoint_config.best_metric}, "
                   f"Save best only: {checkpoint_config.save_best_only}")
    
    # Training loop
    for epoch in range(config.training.epochs):

        train_stats = train_one_epoch(model, model_params, ema_params, train_loader, optimizer, device, epoch, config, logger)
        logger.info(f"Epoch {epoch} loss: {train_stats['loss']}")
        
        # Determine evaluation mode based on warmup phase
        is_warmup = epoch < warmup_epochs
        use_probabilistic_eval = not is_warmup  # Use deterministic during warmup, probabilistic otherwise
        
        eval_mode = "deterministic (cosine similarity)" if is_warmup else "probabilistic (CoPE distance)"
        logger.info(f"Using {eval_mode} evaluation for epoch {epoch}")
        
        # Standard probabilistic evaluation
        eval_stats = evaluate_probabilistic(model, ema_params, val_loader, target_loader, device, config, logger, use_probabilistic=use_probabilistic_eval)
        logger.info(f"Epoch {epoch} eval: {eval_stats}")
        
        # Log CoPE loss parameters (only for non-warmup epochs)
        if epoch >= warmup_epochs and hasattr(model.backbone, 'loss'):
            logger.info(f"Epoch {epoch} CoPE params - "
                       f"shift: {model.backbone.loss.shift.item():.4f}, "
                       f"negative_scale: {model.backbone.loss.negative_scale.item():.4f}")
        
        # Checkpoint saving
        if checkpoint_config:
            should_save, is_best, best_metric_value = should_save_checkpoint(
                epoch, config, eval_stats, best_metric_value, logger)
            
            if should_save:
                try:
                    save_checkpoint_hf(
                        model=model,
                        epoch=epoch,
                        eval_stats=eval_stats,
                        config=config,
                        logger=logger,
                        checkpoint_dir=checkpoint_config.save_dir,
                        is_best=is_best,
                        is_last=False
                    )
                    
                    # Cleanup old checkpoints if max_checkpoints is set
                    if hasattr(checkpoint_config, 'max_checkpoints') and checkpoint_config.max_checkpoints >= 0:
                        cleanup_old_checkpoints(
                            checkpoint_config.save_dir, 
                            checkpoint_config.max_checkpoints, 
                            logger
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at epoch {epoch}: {e}")
    
    # Save final checkpoint if enabled
    if checkpoint_config and getattr(checkpoint_config, 'save_last', True):
        try:
            logger.info("Saving final checkpoint...")
            save_checkpoint_hf(
                model=model,
                epoch=config.training.epochs - 1,
                eval_stats=eval_stats if 'eval_stats' in locals() else {},
                config=config,
                logger=logger,
                checkpoint_dir=checkpoint_config.save_dir,
                is_best=False,
                is_last=True
            )
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    # Parse command line arguments for config file
    parser = argparse.ArgumentParser(description='Training script for probabilistic CoPE models')
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                        help='Path to config file (default: config.yaml)')
    args, unknown = parser.parse_known_args()
    
    # Load base config from specified file
    config = load_config(args.config)

    # Allow command line overrides using OmegaConf
    # Example: python train_probabilistic.py --config my_config.yaml training.batch_size=32 device=cpu
    cli_config = OmegaConf.from_cli(unknown)
    if cli_config:
        config = OmegaConf.merge(config, cli_config)
    
    print(f"Training probabilistic model with config file: {args.config}")
    print("Final config:")
    print(OmegaConf.to_yaml(config))
    
    main(config)
