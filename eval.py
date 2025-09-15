#!/usr/bin/env python3
"""
Evaluation script for CoPE probabilistic models.

This script loads a trained model checkpoint and evaluates it on specified datasets.
Supports both probabilistic and deterministic evaluation modes.

Usage:
    python eval.py --config config-fiq-probabilistic.yaml --checkpoint /path/to/checkpoint
    python eval.py --config config-fiq-probabilistic.yaml --checkpoint /path/to/checkpoint --mode deterministic
    python eval.py --config config-fiq-probabilistic.yaml --checkpoint /path/to/checkpoint
"""

import torch
import logging
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from util.data import build_data
from engine import evaluate_probabilistic
from models import model_registry


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, model, device, logger):
    """
    Load model weights from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory or .bin file
        model: Model instance to load weights into
        device: Device to load model on
        logger: Logger instance
    
    Returns:
        Loaded model and metadata (if available)
    """
    checkpoint_path = Path(checkpoint_path)
    metadata = None
    
    if checkpoint_path.is_dir():
        # HuggingFace format checkpoint directory
        model_file = checkpoint_path / "pytorch_model.bin"
        metadata_file = checkpoint_path / "metadata.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        logger.info(f"Loading model weights from: {model_file}")
        state_dict = torch.load(model_file, map_location=device)
        
        # Load metadata if available
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Checkpoint metadata: epoch {metadata.get('epoch', 'unknown')}, "
                       f"best: {metadata.get('is_best', False)}")
    else:
        # Single .bin file
        logger.info(f"Loading model weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict into model
    model.load_state_dict(state_dict)
    logger.info("✓ Model weights loaded successfully")
    
    return model, metadata


def print_results(eval_stats, logger=None):
    """Print evaluation results in a formatted way"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    # Print recall metrics
    logger.info("Recall Metrics:")
    for metric, value in eval_stats.items():
        if metric.startswith('r_') or 'recall' in metric.lower():
            logger.info(f"  {metric}: {value:.2f}%")
        else:
            logger.info(f"  {metric}: {value:.4f}")
    
    
    logger.info("=" * 60)


def save_results(results_path, eval_stats, metadata=None, config=None):
    """Save evaluation results to JSON file"""
    results = {
        'eval_stats': eval_stats,
        'checkpoint_metadata': metadata,
        'config': OmegaConf.to_container(config, resolve=True) if config else None,
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CoPE probabilistic models')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (directory or .bin file)')
    parser.add_argument('--mode', type=str, choices=['probabilistic', 'deterministic'], 
                        default='probabilistic',
                        help='Evaluation mode (default: probabilistic)')
    parser.add_argument('--no-ema', action='store_true',
                        help='Disable EMA (exponential moving average) parameters')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file to save results (JSON format)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides config)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = OmegaConf.load(args.config)
    
    # Override device if specified
    if args.device:
        config.device = args.device
    
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model: {config.model.name}")
    model = model_registry.get_model(config.model.name, config.model.path).to(device)
    
    # Check if it's a probabilistic model
    is_probabilistic = hasattr(model.backbone, 'loss') or 'prob' in config.model.name.lower()
    if is_probabilistic:
        logger.info("✓ Detected probabilistic model")
    else:
        logger.info("⚠ Non-probabilistic model detected")
    
    # Load checkpoint
    model, checkpoint_metadata = load_checkpoint(args.checkpoint, model, device, logger)
    
    # Build data loaders
    logger.info("Building data loaders...")
    data_components = build_data(config, preprocess=model.preprocess)
    val_loader = data_components['val_loader']
    target_loader = data_components['target_loader']
    
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Target samples: {len(target_loader.dataset)}")
    
    # Set evaluation mode
    use_probabilistic = (args.mode == 'probabilistic') and is_probabilistic
    use_ema = not args.no_ema
    
    eval_mode_str = "probabilistic" if use_probabilistic else "deterministic (cosine similarity)"
    logger.info(f"Evaluation mode: {eval_mode_str}")
    logger.info(f"Using EMA: {use_ema}")
    
    
    # Prepare EMA parameters (use model params if EMA not available)
    ema_params = list(model.parameters()) if use_ema else None
    
    # Run evaluation
    logger.info("Starting evaluation...")
    
    try:
        # Standard evaluation
        eval_stats = evaluate_probabilistic(
            model=model,
            ema_params=ema_params,
            val_loader=val_loader,
            tgt_loader=target_loader,
                device=device,
                config=config,
                logger=logger,
                use_ema=use_ema,
                use_probabilistic=use_probabilistic
            )
        
        # Print results
        print_results(eval_stats, logger)
        
        # Save results if output path specified
        if args.output:
            save_results(args.output, eval_stats, checkpoint_metadata, config)
        
        logger.info("✓ Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()

