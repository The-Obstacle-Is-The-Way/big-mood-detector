import re

with open('src/big_mood_detector/infrastructure/fine_tuning/population_trainer.py', 'r') as f:
    content = f.read()

# Replace the problematic section with properly indented code
old_section = '''        Args:
            sequences: Activity sequences (N, 60)
            labels: Task labels
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Validation split ratio

        Returns:
            Training metrics
        """
        # Extract parameters from kwargs
        sequences = kwargs["sequences"]
        labels = kwargs["labels"]
        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 32)
        learning_rate = kwargs.get("learning_rate", 1e-4)
        validation_split = kwargs.get("validation_split", 0.2)

        logger.info(f"Fine-tuning PAT for {self.task_name}")'''

new_section = '''            Args:
                sequences: Activity sequences (N, 60)
                labels: Task labels
                epochs: Training epochs
                batch_size: Batch size
                learning_rate: Learning rate
                validation_split: Validation split ratio

            Returns:
                Training metrics
            """
            # Extract parameters from kwargs
            sequences = kwargs["sequences"]
            labels = kwargs["labels"]
            epochs = kwargs.get("epochs", 10)
            batch_size = kwargs.get("batch_size", 32)
            learning_rate = kwargs.get("learning_rate", 1e-4)
            validation_split = kwargs.get("validation_split", 0.2)

            logger.info(f"Fine-tuning PAT for {self.task_name}")'''

content = content.replace(old_section, new_section)

# Fix the rest of the method
old_lines = content.split('\n')
fixed_lines = []
in_pat_fine_tune = False
fix_indent = False

for i, line in enumerate(old_lines):
    if 'logger.info(f"Fine-tuning PAT for {self.task_name}")' in line and i > 250:
        fixed_lines.append(line)
        fix_indent = True
    elif fix_indent and '    def save_model(' in line:
        fix_indent = False
        fixed_lines.append('        def save_model(')
    elif fix_indent and line.strip():
        # Add proper indentation
        fixed_lines.append('            ' + line.lstrip())
    else:
        fixed_lines.append(line)

content = '\n'.join(fixed_lines)

# Now fix save_model method
content = content.replace(
    '    ) -> Path:\n        """Save fine-tuned model.',
    '        ) -> Path:\n            """Save fine-tuned model.'
)

# Fix save_model docstring and body
old_save = '''        """Save fine-tuned model.

        Args:
            encoder: PAT encoder
            task_head: Task-specific head
            task_name: Task name
            metrics: Training metrics

        Returns:
            Path to saved model
        """
        # Save PyTorch model
        model_path = self.output_dir / f"pat_{task_name}.pt"'''

new_save = '''            """Save fine-tuned model.

            Args:
                encoder: PAT encoder
                task_head: Task-specific head
                task_name: Task name
                metrics: Training metrics

            Returns:
                Path to saved model
            """
            # Save PyTorch model
            model_path = self.output_dir / f"pat_{task_name}.pt"'''

content = content.replace(old_save, new_save)

# Fix the rest of save_model
lines = content.split('\n')
fixed_lines = []
in_save_model_body = False

for i, line in enumerate(lines):
    if 'model_path = self.output_dir / f"pat_{task_name}.pt"' in line and i > 350:
        fixed_lines.append(line)
        in_save_model_body = True
    elif in_save_model_body and 'class XGBoostPopulationTrainer' in line:
        in_save_model_body = False
        fixed_lines.append(line)
    elif in_save_model_body and line.strip() and not line.strip().startswith('else:'):
        # Ensure proper indentation
        stripped = line.lstrip()
        if not stripped.startswith('    '):
            fixed_lines.append('            ' + stripped)
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

with open('src/big_mood_detector/infrastructure/fine_tuning/population_trainer.py', 'w') as f:
    f.write('\n'.join(fixed_lines))

print("Fixed indentation")