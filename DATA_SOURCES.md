# Data Sources and Attribution

This document provides complete attribution for all datasets used in this office item classification project.

## Dataset Collection Summary

**Collection Date:** October 9-10, 2025  
**Total Classes:** 11  
**Total Images:** 13,616  
**Source Platform:** Roboflow Universe  
**Download Format:** COCO JSON  
**Download Method:** Manual download from Roboflow Universe

## Individual Dataset Details

### 1. Computer Mouse
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/4-52p2c/office-mouse
- **Workspace:** 4-52p2c
- **Project:** office-mouse
- **Images Collected:** 724
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 9, 2025

### 2. Keyboard
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/alom/all-keyboard
- **Workspace:** alom
- **Project:** all-keyboard
- **Images Collected:** 811
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 9, 2025

### 3. Stapler
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/ashley-t0x9a/stapler-rdudj
- **Workspace:** ashley-t0x9a
- **Project:** stapler-rdudj
- **Version:** 12
- **Images Collected:** 1,354
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 9, 2025

### 4. Laptop
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/project-one-ejtx6/laptop-detection-
- **Workspace:** project-one-ejtx6
- **Project:** laptop-detection
- **Version:** 4
- **Images Collected:** 1,547
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 9, 2025

### 5. Office Chair
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/project-yoa2s/chair-bgkdb
- **Workspace:** project-yoa2s
- **Project:** chair-bgkdb
- **Version:** 1
- **Images Collected:** 777
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 9, 2025

### 6. Mug
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/my-workspace-7j2fi/cup-detection-w8kfb/dataset/2
- **Workspace:** my-workspace-7j2fi
- **Project:** cup-detection-w8kfb
- **Version:** 2
- **Images Collected:** 794
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 9, 2025
- **Note:** Includes both cups and mugs

### 7. Pen
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/real-time-object-detection-navigation/pen-detection-lexd8/dataset/1
- **Workspace:** real-time-object-detection-navigation
- **Project:** pen-detection-lexd8
- **Version:** 1
- **Images Collected:** 915
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 9, 2025

### 8. Notebook
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/study-group/notebook-eiabn/dataset/2
- **Workspace:** study-group
- **Project:** notebook-eiabn
- **Version:** 2
- **Images Collected:** 2,591 (reduced to 1,500)
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 9, 2025
- **Note:** Randomly sampled 1,500 images from original 2,591 for dataset balance

### 9. Mobile Phone
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/testlabel-6jpzh/cell-phone-detect-fnmnk/dataset/1
- **Workspace:** testlabel-6jpzh
- **Project:** cell-phone-detect-fnmnk
- **Version:** 1
- **Images Collected:** 1,670
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 10, 2025

### 10. Office Bin
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/test-fg7sa/bin-detection-test2/dataset/1
- **Workspace:** test-fg7sa
- **Project:** bin-detection-test2
- **Version:** 1
- **Images Collected:** 1,668
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train, valid, test
- **Date Downloaded:** October 10, 2025

### 11. Water Bottle
- **Source:** Roboflow Universe
- **Project URL:** https://universe.roboflow.com/can-jniv8/water-bottle-eovbe
- **Workspace:** can-jniv8
- **Project:** water-bottle-eovbe
- **Version:** 21
- **Images Collected:** 1,356
- **License:** As specified on Roboflow project page
- **Format:** COCO JSON
- **Splits Used:** train only (no valid/test splits in original)
- **Date Downloaded:** October 10, 2025

## Data Processing Pipeline

1. **Download:** Manual download of each dataset from Roboflow Universe in COCO JSON format
2. **Extraction:** Extracted downloaded .zip files to Downloads folder
3. **Organization:** Copied all images (from train, valid, and test splits) into single class folders:
   - Combined all splits into `data/raw/[class_name]/` directories
   - Original split information discarded at this stage
4. **Quality Control:** Manual review of dataset sources before download
5. **Balancing:** Reduced notebook class from 2,591 to 1,500 images (random sampling)
6. **Future Split:** Will be split into train/val/test (70/15/15) using `organize_dataset.py`

## Dataset Statistics

| Class          | Images | Percentage |
|----------------|--------|------------|
| Laptop         | 1,547  | 11.4%      |
| Mobile Phone   | 1,670  | 12.3%      |
| Office Bin     | 1,668  | 12.3%      |
| Notebook       | 1,500  | 11.0%      |
| Water Bottle   | 1,356  | 10.0%      |
| Stapler        | 1,354  | 9.9%       |
| Pen            | 915    | 6.7%       |
| Keyboard       | 811    | 6.0%       |
| Mug            | 794    | 5.8%       |
| Office Chair   | 777    | 5.7%       |
| Computer Mouse | 724    | 5.3%       |
| **TOTAL**      | **13,616** | **100%** |

## Ethical Considerations

- All datasets sourced from public Roboflow Universe projects
- Each dataset has its own license specified on the project page
- Data used solely for educational/academic purposes (PDE3802 module assessment)
- No personal or sensitive data present in images
- Proper attribution provided for all sources
- Images will not be redistributed; code and documentation provided instead

## Licensing Notes

- Each dataset retains its original license from Roboflow Universe
- Users of this project should verify licenses before any commercial use
- This project follows academic fair use guidelines
- Dataset creators are acknowledged and attributed appropriately

## Acknowledgments

We gratefully acknowledge the following contributors to Roboflow Universe who made their datasets publicly available:
- 4-52p2c (Office Mouse)
- alom (All Keyboard)
- ashley-t0x9a (Stapler)
- project-one-ejtx6 (Laptop Detection)
- project-yoa2s (Chair)
- my-workspace-7j2fi (Cup Detection)
- real-time-object-detection-navigation (Pen Detection)
- study-group (Notebook)
- testlabel-6jpzh (Cell Phone Detect)
- test-fg7sa (Bin Detection)
- can-jniv8 (Water Bottle)

## Citation

If using this dataset collection for research, please cite:

```
@misc{office_item_classifier_2025,
  author = {Oluwatunmise Shuaibu Raphael},
  title = {Office Item Classification Dataset},
  year = {2025},
  publisher = {Middlesex University London},
  note = {PDE3802 Assessment - AI in Robotics}
}
```

Additionally, please cite individual Roboflow Universe projects using their provided citation information.

## Contact

For questions about this dataset compilation:
- Student: Oluwatunmise Shuaibu Raphael
- Student ID: M00960413
- Module: PDE3802 - AI in Robotics
- Institution: Middlesex University London

---

*Last Updated: October 10, 2025*