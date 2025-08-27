"""Main training script for the activation comparison study."""
import argparse
import uuid
from models.activations import activation_names
from train.seed import set_seed

def main():
    parser = argparse.ArgumentParser(description="Activation Comparison Study Trainer")
    
    # Task and model configuration
    parser.add_argument("--task", type=str, required=True, choices=["vision", "cls", "lm"],
                        help="Task to run (vision, cls, lm).")
    parser.add_argument("--activation", type=str, required=True, choices=activation_names(),
                        help="Activation function to use.")
    
    # Run identification and logging
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--run_id", type=str, default=str(uuid.uuid4()),
                        help="Unique identifier for the run.")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs and artifacts.")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print("--- Training Configuration ---")
    print(f"Task: {args.task}")
    print(f"Activation: {args.activation}")
    print(f"Seed: {args.seed} (applied)")
    print(f"Run ID: {args.run_id}")
    print(f"Log Directory: {args.log_dir}")
    print("--------------------------")
    
    # Placeholder for PyTorch Lightning trainer
    # trainer = pl.Trainer(...)
    # trainer.fit(...)
    
    print("\nCLI parsing and seeding complete. Ready for PyTorch Lightning integration.")

if __name__ == "__main__":
    main()
