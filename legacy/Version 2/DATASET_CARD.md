# Dataset Card: Office Items Classification Dataset

## Dataset Description

**Version:** 2.0 (In Progress)  
**Date Created:** October 10, 2025  
**Last Updated:** October 10, 2025  
**Purpose:** Improved real-world performance for office item classification  
**Total Images:** TBD  
**Total Classes:** 11  
**Status:** 🔄 Data collection in progress

---

## Dataset Summary

Version 2.0 focuses on bridging the domain gap between clean training data and real-world deployment conditions. This dataset prioritizes diversity, realism, and representation of actual office environments to achieve robust performance in practical applications.

### Motivation for Version 2.0

**Version 1.0 Limitations:**
- Training images: Isolated objects, plain backgrounds, perfect lighting
- Real-world performance: ~60-70% camera accuracy despite 96.37% test accuracy
- Domain gap: Clean web-scraped images did not generalize to messy environments

**Version 2.0 Goals:**
- Achieve 85-90% accuracy in real-world camera conditions
- Include diverse backgrounds, lighting, and viewing angles
- Represent actual office environments and use cases
- Minimize domain gap between training and deployment

---

## Classes (11 total)

1. **Computer Mouse** - Wired and wireless mice
2. **Keyboard** - Various keyboard types and sizes
3. **Laptop** - Laptop computers in various states
4. **Mobile Phone** - Smartphones and mobile devices
5. **Mug** - Coffee mugs, tea cups, drinking vessels
6. **Notebook** - Paper notebooks, journals, notepads
7. **Office Bin** - Waste bins, recycling bins
8. **Office Chair** - Desk chairs, office seating
9. **Pen** - Pens, pencils, writing instruments
10. **Stapler** - Office staplers
11. **Water Bottle** - Reusable water bottles

---

## Data Collection Strategy

### Quality Criteria

**Required for Each Image:**
- Clear visibility of the target object
- Various viewing angles (not just frontal)
- Diverse backgrounds (desks, shelves, hands, in-use)
- Multiple lighting conditions (natural, artificial, mixed)
- Real office contexts

**Diversity Goals:**
- **Backgrounds:** Plain (20%), cluttered desk (40%), in-use/hand (40%)
- **Lighting:** Bright (30%), normal (50%), dim (20%)
- **Angles:** Frontal (30%), side (30%), top-down (20%), angled (20%)
- **Context:** Isolated (30%), with other objects (70%)

### Target Distribution

Aim for balanced dataset:
- **Minimum per class:** 800 images
- **Target per class:** 1,000-1,500 images
- **Total target:** 11,000-16,500 images

---

## Data Sources

**Status:** Collection in progress

Sources will be documented in `DATA_SOURCES.md` as collection proceeds.

**Planned Sources:**
- High-quality Roboflow Universe datasets (filtered for diversity)
- Custom photography (real office environments)
- Public domain office image repositories
- Potentially: Synthetic data augmentation for rare angles

**Selection Criteria:**
- Image quality and resolution
- Background diversity
- Angle variety
- Lighting variation
- Context realism

---

## Data Format

**Image Specifications:**
- **Formats:** JPEG, PNG
- **Storage:** `data/raw/[class_name]/[image_name].[ext]`
- **Input Size:** 224×224 pixels (resized during training)
- **Color:** RGB (3 channels)

**Normalization:**
- **Mean:** [0.485, 0.456, 0.406] (ImageNet statistics)
- **Std:** [0.229, 0.224, 0.225] (ImageNet statistics)

---

## Train/Validation/Test Split

**Planned Split Ratios:**
- **Training:** 70%
- **Validation:** 15%
- **Testing:** 15%

**Split Method:**
- Stratified sampling to maintain class distribution
- Random seed 42 for reproducibility
- Performed by `src/organize_dataset.py`

**Output Structure:**
```
data/processed/
├── train/
│   ├── computer_mouse/
│   ├── keyboard/
│   └── ...
├── val/
│   ├── computer_mouse/
│   └── ...
└── test/
    ├── computer_mouse/
    └── ...
```

---

## Data Augmentation

### Training Augmentation (Applied During Training)

**Geometric:**
- Random resize and crop to 224×224
- Random horizontal flip (50% probability)
- Random rotation (±15 degrees)

**Photometric:**
- Color jitter:
  - Brightness: ±20%
  - Contrast: ±20%
  - Saturation: ±20%
- Random gaussian blur (optional)

**Normalization:**
- ImageNet mean and standard deviation

### Validation/Test (No Augmentation)
- Resize to 256×256
- Center crop to 224×224
- ImageNet normalization only

---

## Quality Control

**Pre-Collection:**
- Manual review of source datasets
- Quality assessment criteria checklist
- Diversity verification

**Post-Collection:**
- Manual inspection of samples
- Duplicate detection and removal
- Quality filtering (blur, corruption)
- Balance verification across classes

**Acceptance Criteria:**
- Minimum 800 images per class
- <3:1 ratio between largest and smallest class
- No duplicate or near-duplicate images
- <5% low-quality images (blur, corruption)

---

## Evaluation Metrics

**Model Performance Metrics:**
- Overall accuracy
- Macro F1-score (average across all classes)
- Per-class precision, recall, F1-score
- Confusion matrix analysis

**Domain Gap Metrics:**
- Test set accuracy vs. real-world camera accuracy
- Performance across different backgrounds
- Performance across lighting conditions
- Robustness to viewing angles

**Target:** <10% gap between test accuracy and real-world accuracy

---

## Ethical Considerations

**Privacy:**
- No personal or sensitive data in images
- No identifiable faces or private information
- Office items only

**Attribution:**
- All sources properly cited in DATA_SOURCES.md
- Licenses verified before use
- Original creators acknowledged

**Usage:**
- Educational/academic purposes only (PDE3802 assessment)
- Not for commercial distribution
- Code and methodology shared, not raw images

**Fairness:**
- Diverse representation of object types
- No bias toward specific brands or styles
- Inclusive of various office environments

---

## Known Limitations

**Current (Anticipated):**
- Limited to 11 common office items
- May not generalize to rare/unusual variants
- Performance depends on lighting quality
- Optimal for desktop/handheld items (not large furniture beyond chairs)

**Mitigation Strategies:**
- Comprehensive data collection across conditions
- Extensive validation testing
- Clear documentation of limitations
- Confidence thresholding for deployment

---

## Version History

**Version 1.0 (Archived):**
- **Sources:** Roboflow Universe (11 projects)
- **Size:** 13,616 images
- **Test Accuracy:** 96.37%
- **Real-world Accuracy:** ~60-70%
- **Limitation:** Domain gap between training and deployment
- **Documentation:** `legacy/docs_v1/dataset_card_v1.md`

**Version 2.0 (Current):**
- **Focus:** Bridge domain gap
- **Strategy:** Diverse, realistic data
- **Goal:** 85-90% real-world accuracy
- **Status:** Collection in progress

---

## Citation

```bibtex
@misc{office_items_dataset_v2_2025,
  title={Office Items Classification Dataset Version 2.0},
  author={Raphael, Oluwatunmise Shuaibu},
  year={2025},
  institution={Middlesex University London},
  note={PDE3802 Assessment - Improved dataset for real-world deployment}
}
```

---

## Contact & Support

**Student:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413  
**Module:** PDE3802 - AI in Robotics  
**Institution:** Middlesex University London  
**Academic Year:** 2025-26

---

*This dataset card will be updated as data collection progresses.*  
*Last Updated: October 10, 2025*