# Office Item Classifier

**Module:** PDE3802 - AI in Robotics  
**Assessment:** Part B - Office-Goods Classification Code  
**Author:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413

## Project Overview

An image classification system that recognizes common office items from single images or live camera feed. This project uses deep learning (PyTorch with transfer learning) to classify 10 different office object categories.

## Classes to Recognize

1. Mug
2. Water Bottle
3. Mobile Phone
4. Keyboard
5. Computer Mouse
6. Stapler
7. Pen/Pencil
8. Notebook
9. Office Chair
10. Office Bin

## Project Structure

```
office-item-classifier/
├── data/
│   ├── raw/                # Original downloaded images
│   ├── processed/          # Organized train/val/test splits
│   └── dataset_card.md     # Dataset documentation
├── src/
│   └── (classification code will go here)
├── models/
│   └── (trained models will be saved here)
├── notebooks/
│   └── (exploration and training notebooks)
├── results/
│   └── (confusion matrices, metrics, etc.)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd office-item-classifier

# Install dependencies
pip install -r requirements.txt
```

## Usage

*Coming soon - code under development*

## Development Stages

- [x] Project structure setup
- [ ] Dataset collection and organization
- [ ] Data preprocessing and augmentation
- [ ] Model selection and training
- [ ] Evaluation and testing
- [ ] Inference script (file and camera input)
- [ ] Documentation and video walkthrough

## Requirements

See `requirements.txt` for full dependencies.

## License

MIT License - Academic Project

## Acknowledgments

- Middlesex University London
- PDE3802 Module Team