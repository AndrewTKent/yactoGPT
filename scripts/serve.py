# scripts/serve.py
#!/usr/bin/env python3
"""Run inference server for yacto-GPT."""

import os
import sys
import argparse
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import InferenceServer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run yacto-GPT inference server')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/inference.yaml',
        help='Path to inference config'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to'
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override host/port if provided
    config['api']['host'] = args.host
    config['api']['port'] = args.port
    
    # Create and run server
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Loading model from {args.checkpoint}...")
    
    server = InferenceServer(
        model_path=args.checkpoint,
        config=config
    )
    
    print(f"Server ready! Access at http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /generate")
    print("  POST /generate/stream")
    
    server.run()


if __name__ == '__main__':
    main()