# Data Sources and Attribution

## Dataset Collection Summary

**Version:** 2.0  
**Collection Date:** October 10, 2025 - TBD  
**Status:** 🔄 Collection in progress  
**Total Classes:** 11  
**Total Images:** TBD  
**Source Platform:** TBD

---

## Data Collection Strategy

### Objectives

**Primary Goals:**
1. Bridge domain gap between training and real-world deployment
2. Achieve 85-90% accuracy in real-world camera conditions
3. Include diverse backgrounds, lighting, and viewing angles
4. Represent actual office environments

### Quality Criteria

**Each source must provide:**
- High-resolution images (minimum 224×224, prefer larger)
- Diverse viewing angles (not just frontal shots)
- Various backgrounds (not just plain/white)
- Multiple lighting conditions
- Real-world context and usage scenarios

---

## Individual Dataset Details

**Format for documentation:**

```
### Class: [Class Name]
- **Source:** [Platform/Provider]
- **Project URL:** [Link]
- **Images Collected:** [Number]
- **License:** [License Type]
- **Date Downloaded:** [Date]
- **Quality Assessment:** [Brief notes on diversity, quality]
- **Rationale:** [Why this source was selected]
```

---

## Planned Sources

### Potential Platforms

**Primary:**
- Roboflow Universe (filtered for high-quality, diverse datasets)
- Custom photography (real office environments)
- Open Images Dataset (Google)
- ImageNet subsets

**Secondary (if needed):**
- COCO dataset filtered subsets
- Public domain image repositories
- Creative Commons licensed collections

### Selection Process

**Step 1: Source Identification**
- Search platforms for each office item class
- Preview dataset samples for quality assessment
- Check diversity (backgrounds, angles, lighting)

**Step 2: Quality Verification**
- Manual review of sample images (minimum 20 per source)
- Check for:
  - Image resolution and clarity
  - Background variety
  - Angle diversity
  - Lighting conditions
  - Context realism
- Verify licensing permits educational use
- Assess total available images per class

**Step 3: Download and Organization**
- Download in appropriate format
- Organize into `data/raw/[class_name]/` structure
- Document source details immediately
- Note any preprocessing requirements

**Step 4: Quality Control**
- Remove duplicates
- Filter low-quality images (blur, corruption)
- Verify class labels
- Check for adequate diversity

---

## Dataset Statistics

**Will be documented here after collection:**

| Class          | Images | Source Count | Notes |
|----------------|--------|--------------|-------|
| Computer Mouse | TBD    | TBD          | TBD   |
| Keyboard       | TBD    | TBD          | TBD   |
| Laptop         | TBD    | TBD          | TBD   |
| Mobile Phone   | TBD    | TBD          | TBD   |
| Mug            | TBD    | TBD          | TBD   |
| Notebook       | TBD    | TBD          | TBD   |
| Office Bin     | TBD    | TBD          | TBD   |
| Office Chair   | TBD    | TBD          | TBD   |
| Pen            | TBD    | TBD          | TBD   |
| Stapler        | TBD    | TBD          | TBD   |
| Water Bottle   | TBD    | TBD          | TBD   |
| **TOTAL**      | **TBD** | **TBD**     | -     |

---

## Data Processing Pipeline

### Collection Phase
1. Identify and verify sources
2. Download raw images
3. Organize into class folders
4. Document source attribution

### Quality Control Phase
1. Manual sample inspection
2. Duplicate detection and removal
3. Quality filtering (resolution, clarity)
4. Balance verification

### Organization Phase
1. Combine all sources per class
2. Verify minimum image counts
3. Run `organize_dataset.py` for train/val/test split
4. Validate split distributions

### Documentation Phase
1. Update this file with complete source details
2. Update dataset card with statistics
3. Create visual samples for verification
4. Archive source information

---

## Ethical Considerations

**Privacy:**
- No personal identifiable information
- No private or sensitive content
- Office items only, no people

**Licensing:**
- Verify all licenses permit educational use
- Respect attribution requirements
- No commercial redistribution

**Attribution:**
- Properly cite all original creators
- Maintain license compliance
- Provide source links where applicable

**Fair Use:**
- Educational/academic purpose only
- Non-commercial assessment use
- Following university guidelines

---

## Quality Metrics

**Target Metrics:**
- **Minimum images per class:** 800
- **Class balance ratio:** <3:1 (largest to smallest)
- **Diversity score:** High (varied backgrounds, angles, lighting)
- **Quality score:** >95% usable images

**Diversity Breakdown Goals:**
- Backgrounds: 20% plain, 40% desk/shelf, 40% in-use/complex
- Lighting: 30% bright, 50% normal, 20% dim
- Angles: 30% frontal, 30% side, 20% top, 20% angled
- Context: 30% isolated, 70% with other objects

---

## Version History

### Version 1.0 (Archived)

**Source:** Roboflow Universe  
**Projects:** 11 different datasets  
**Total Images:** 13,616  
**Date:** October 9-10, 2025

**Detailed Attribution:** See `legacy/docs_v1/DATA_SOURCES_v1.md`

**Limitations Identified:**
- All images web-sourced (Roboflow)
- Too clean and isolated
- Plain backgrounds dominated
- Limited angle diversity
- Poor real-world generalization

**Lessons Learned:**
- Need diverse backgrounds
- Multiple viewing angles critical
- Lighting variation important
- Real context improves generalization

---

### Version 2.0 (Current)

**Status:** Collection in progress  
**Strategy:** Address v1.0 limitations  
**Documentation:** Will be completed during collection phase

**Improvements:**
- Prioritize background diversity
- Include multiple viewing angles
- Ensure lighting variation
- Seek real office contexts
- Mix web sources with custom photos (if applicable)

---

## Acknowledgments

Acknowledgments for dataset creators and platforms will be added here as sources are finalized.

**Platforms to be acknowledged:**
- Dataset hosting platforms used
- Individual dataset creators
- Open source communities
- Image repositories

---

## Contact

**Student:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413  
**Module:** PDE3802 - AI in Robotics  
**Institution:** Middlesex University London

For questions about data sources or collection methodology, refer to the student contact above.

---

*This document will be populated with complete source details as data collection progresses.*  
*Last Updated: October 10, 2025*  
*Status: Template prepared, awaiting data collection*