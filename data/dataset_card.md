# Dataset Card: Office Items Classification Dataset

## Dataset Description

**Version:** 1.0  
**Date Created:** October 2025  
**Purpose:** Training and evaluation of office item classification model for PDE3802 assessment

## Dataset Summary

This dataset contains images of 10 common office items collected for the purpose of training an autonomous office organizing robot's perception system.

### Classes (10 total)

1. **Mug** - Coffee mugs, tea cups
2. **Water Bottle** - Reusable water bottles
3. **Mobile Phone** - Smartphones
4. **Keyboard** - Computer keyboards (wired/wireless)
5. **Computer Mouse** - Computer mice (wired/wireless)
6. **Stapler** - Office staplers
7. **Pen/Pencil** - Writing instruments
8. **Notebook** - Paper notebooks, journals
9. **Office Chair** - Desk chairs, office seating
10. **Office Bin** - Waste bins, recycling bins

## Data Sources

### Initial Dataset (Downloaded)
- **Source:** [To be documented]
- **Download Date:** [To be added]
- **License:** [To be verified]
- **Number of images per class:** [To be counted]

### Custom Dataset (Self-captured)
- **Capture Date:** [To be added]
- **Equipment:** [Camera/phone specifications]
- **Number of images per class:** [To be added]

## Dataset Statistics

| Split      | Total Images | Images per Class (avg) |
|------------|--------------|------------------------|
| Training   | TBD          | TBD                    |
| Validation | TBD          | TBD                    |
| Testing    | TBD          | TBD                    |
| **Total**  | **TBD**      | **TBD**                |

## Data Collection Methodology

1. **Downloaded Images:**
   - Sourced from public datasets
   - Filtered for quality and relevance
   - Verified class labels

2. **Self-captured Images:**
   - Multiple angles per object
   - Various lighting conditions
   - Different backgrounds
   - Real office environment

## Data Processing

- Image format: JPEG/PNG
- Resized to: 224x224 pixels (for model input)
- Normalization: ImageNet statistics
- Augmentation: [To be documented during training]

## Quality Control

- Manual verification of labels
- Removal of corrupted/invalid images
- Consistent image quality standards

## Ethical Considerations

- No personal or sensitive data in images
- Public spaces or personal items only
- Proper attribution of source datasets

## Limitations

- Limited variety in some classes
- Potential bias toward certain object styles
- Indoor lighting conditions primarily

## Updates

- **v1.0** (Oct 2025): Initial dataset compilation
- Future versions will include additional self-captured images

---

*This dataset card will be updated as data collection and processing progresses.*