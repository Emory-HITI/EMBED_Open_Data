# EMBED_Open_Data
Data descriptor and sample notebooks for the Emory Breast Imaging Dataset (EMBED) hosted on the AWS Open Data Program
**EMBED Overview:**  

**I. Summary**

EMBED contains 364,000 screening and diagnostic mammographic exams for 110,000 patients from four hospitals over an 8-year period. The EMBED AWS Open Data release represents 20% of the dataset divided into two equal cohorts at the patient level. This release of the dataset includes 2D and C-view images. Digital breast tomosynthesis, ultrasound, and MRI exams will be added at a later date.

**II. Primer on Mammography**

A full discussion on screening mammography is beyond the scope of this documentation, however we will provide a brief primer. In the United States, women are recommended annual screening mammography beginning at age 40 and biennially after age 55. Screening mammograms are recommended for **asymptomatic** patients to detect occult breast cancer. Approximately 90% of screeninng mammograms are normal, and assigned a BIRADS 1 (negative) or BIRADS 2 (benign) after which the woman will return to screening in 1-2 years. However, approximately 10% of screening exams demonstrate an abnormality that requires further imaging and are assigned BIRADS 0 (additional evaluation). BIRADS 3 is a special case where there is low suspicion and the woman will return for screening again in 6 months.

If a woman is assigned BIRADS 0, she will proceed to diagnostic mammogram with special views including compression and magnification paddles. She may also undergo concurrent ultrasound. In 2/3 of diagnostic mammograms, the finding disappears or is resolved as benign and the woman will be assigned BIRADS 1 or 2. In 1/3 of diagnostic mammograms, the finding persists and the woman is assigned a BIRADS score of 4 (suspicious) or 5 (highly suspicious) and will proceed to biopsy. Approximately 2/3 of biopsy results are benign and the woman will return to annual screening. Approximately 1/3 of biopsy results are malignant in which case she will go for further imaging (MRI to evaluate extent of disease) and/or surgical resection (lumpectomy or mastectomy). The patient will receive diagnostic mammograms for several years after this before returning to annual screening.

Lastly, **symptomatic** patients do not receive screening mammography. Any patient with pain, discharge, or other symptoms would go straight to diagnostic mammography where they will be evaluated, assigned a BIRADS score, and proceed to biopsy or further imaging if indicated. In these cases, patients may have a pathology result linked to a diagnostic exam that is NOT preceeded by a screening exam.

**III. Summary Statistics**
   1. **Total Patients**: 22,382
   1. **Total Exams**: 76,373
      1. Screening: 72% 
      1. Diagnostic: 28% 
   1. **Total Images**: 480,323
      1. 2D: 76%
      1. C-view: 24%
   1. **Manufacturers**
      1. Hologic: 92%
      1. GE: 6%
      1. Fujifilm: 2%
   1. **Race**: 
      1. White: 40% White
      1. African American 40%
      1. Asian: 6.5% 
   1. **Ethnicity**: 5.6% Hispanic

**IV. File Structure**

The dataset is structured as **./cohort/patient\_ID/study\_ID/SOPinstanceUID.dcm**

Each cohort contains 11,000 unique patients and **all** exams for those patients. There is no overlap in patients or exams between cohorts 1 and 2. All exams over time for a patient will be contained in the same **./cohort/patient\_ID/** folder. Similarly, all images for an exam will be contained in the same **./cohort/patient\_ID/study\_ID/** folder. 

**V. Data Tables**
   1. Data tables are located in the **./tables** subfolder
   1. **EMBED\_OpenData\_clinical.csv** contains clinical information for exams and is structured as one row *per* *finding* as described by the radiologist. Information includes exam type, imaging descriptors (mass, calcification), laterality of findings, pathology results, and patient demographics. Detailed information is in section VI.
   1. **EMBED\_OpenData\_metadata.csv** contains image level information and is structured as one row *per file*. Information includes DICOM metadata, ROI location, image type (2D, 3D, c-view), and file path. Detailed information in section VII.
   1. **EMBED\_OpenData\_clinical\_reduced.csv** and **EMBED\_OpenData\_metadata\_reduced.csv** contain reduced numbers of fields that represent the most frequently used information from each file. This may be easier to work with for beginners and faster to load.

**VI. Clinical Data**

The file **EMBED\_OpenData\_clinical.csv** contains all clinical data and demographics data collected for an exam, as detailed below. In this file, **each row represents a single finding** on mammography, and therefore there can be 1 to several rows per exam. These data are entered by the radiologist at the time of interpretation. Pathology information is entered retroactively by an administrator for all patients who receive breast biopsies, lumpectomies, or mastectomies and are attributed to the original finding that led to the biopsy/surgery.  

1. **Data Types**:
   1. **Findings/descriptors** - generated by the radiologist at the time of interpretation and saved in a structured format. Includes information about masses, calcifications, asymmetries, etc on a **per finding** basis. If a given exam has multiple findings (for example one lesion in the left breast and one in the right), each row will contain information regarding one finding. BIRADS scores are also assigned on a per finding basis. See examples below.
   1. **Pathology outcomes** - manually entered retroactively an administrator and linked to individual findings. Pathology results are entered **into the corresponding row of the specific findings that was biopsied or resected**. See examples below.
   1. **Global information**  - this includes included study descriptions, visit type, and patient demographics. Since these data are independent of individual findings, they are repeated in each row for a given exam regardless of the number of findings.

1. **Examples**
   1. Negative screening mammogram – this exam be represented by a single row without any population of findings in the mass or calcification fields and without any pathology information
   1. Screening mammogram requiring additional evaluation of the left breast - this will be represented by two rows. One row will represent additional evaluation required (BIRADS – 0) of the left breast along with imaging findings described by the radiologist, and a second row will represent negative assessment of the right breast (BIRADS – 1).
   1. Left sided diagnostic mammogram with two findings, one of which was biopsied - this will be represented by two rows. The first row will represent the first abnormal finding in the left breast along with corresponding imaging findings and pathology information. The second row will represent the second finding in the left breast with corresponding imaging findings but no pathology information because the finding was not biopsied.

**Key fields:**

A full legend can be found in **AWS_Open_Data_Clinical_Legend.csv**

|**Feature name** |**Description** |
| - | - |
|empi\_anon|Unique anonymized patient ID, all exams for a patient will have the same ID|
|acc\_anon |Unique ID per exam, all rows for an exam will have the same acc\_anon (and the same empi\_anon). Negative exams or exams with only one finding will have a single row per acc\_anon, and exams with multiple findings will have multiple rows.|
|study\_date\_anon|Anonymized date that the exam was signed. This may differ slightly than the date the exam was acquired. All dates are shifted randomly across patients, but the same *within* a patient to maintain temporality between multiple exams and pathology results for a patient.|
|desc |The study description such as screening or diagnostic mammogram|
|tissueden|<p>BIRADS breast density</p><p>**1**: The breasts are almost entirely fat (BIRADS A) </p><p>**2**: Scattered fibroglandular densities (BIRADS B)    </p><p>**3**: Heterogeneously dense (BIRADS C) </p><p>**4**: Extremely dense (BIRADS D) </p><p>**5**: Normal male**   </p>|
|asses|<p>The [BI-RADS score](https://www.acr.org/-/media/ACR/Files/RADS/BI-RADS/Mammography-Reporting.pdf) of the exam. This is assigned globally for an exam but repeated in each finding row. </p><p>BIRADS 0: **A** – Additional evaluation</p><p>BIRADS 1: **N** – Negative </p><p>BIRADS 2: **B** - Benign</p><p>BIRADS 3: **P** – Probably benign </p><p>BIRADS 4: **S** – Suspicious </p><p>BIRADS 5: **M**- Highly suggestive of malignancy </p><p>BIRADS 6: **K** - Known biopsy proven </p><p></p><p>Screening exams may have BIRADS 0, 1, 2, or 3. Diagnostic exams may have BIRADS 4, 5, or 6.</p>|
|numfind|Index of the finding number for an exam beginning with **1**|
|side|<p>Side of the finding described in the current row</p><p>**L:** left</p><p>**R**: right</p><p>**B**: bilateral </p>|
|total\_l\_find|Number of unique findings for the left breast for a given exam. |
|total\_r\_find|Number of unique findings for the right breast for a given exam.|
|massshape |Mass shape according to BIRADS descriptors.  Also includes asymmetries and architectural distortion (see ./tables/clinical\_legend.csv)|
|massmargin|Mass margin according to BIRADS descriptors (see ./tables/clinical\_legend.csv)|
|massdens|Mass density according to BIRADS descriptors (see ./tables/clinical\_legend.csv)|
|calcfind|Type of calcification according to BIRADS descriptors (see ./tables/clinical\_legend.csv)|
|calcdistri|Distribution of calcifications according to BIRADS descriptors (see ./tables/clinical\_ legend.csv) |
|bside|<p>Laterality of any pathology result</p><p>**L**: left </p><p>**R**: right</p>|
|procdate_anon|Date of pathology result. |
|type|Source of the of tissue specimen obtained - biopsy, FNA, lumpectomy, etc. (see ./tables/clinical\_legend.csv). This is helpful in cases where this is a biopsy followed by a lumpectomy. The pathology entries will contain information from both events, however the lumpectomy pathology results would typically supersede biopsy results|
|path1- path10|Individual pathologic diagnoses from a given specimen. For example, a given specimen may contain invasive ductal carcinoma (ID), ductal carcinoma in situ (DC), and radial scar (RS). This row would contain these entries in path 1 – path 3 (see ./tables/clinical\_legend.csv)|
|path\_severity|<p>The most severe pathology result from a given specimen, abstracted from path1 – path10 (see see ./tables/pathology\_legend.csv for classification schema)</p><p>**0**: invasive cancer</p><p>**1**: non-invasive cancer</p><p>**2**: high-risk lesion</p><p>**3**: borderline lesion</p><p>**4**: benign findings</p><p>**5**: negative (normal breast tissue)</p><p>**6**: non-breast cancer</p>|
|RACE\_DESC|Patient Race|
|ETHNIC\_GROUP\_DESC|Patient Ethnicity|


**VII. Metadata**

Contains image level information and is structured as one row per file. Information includes DICOM metadata, ROI location, image type (2D, 3D, c-view), and file path. 

|**Feature name** |**Description** |
| - | - |
|empi\_anon|Unique anonymized patient ID, all exams for a patient will have the same ID|
|acc\_anon |Unique ID per exam, all images within an exam will have the same acc\_anon (and the same empi\_anon)|
|anon\_dicom\_path|Anonymized full dicom file path|
|png\_path|Full png image path (*not relevant for AWS Open Data release*)|
|png\_filename|Filename only of the png  (*not relevant for AWS Open Data release*). Filenames are hashed and therefore are unique across all files, allowing this field to be used as an index if desired.|
|study\_date\_anon|Anonymized date of *acquisition* of the exam. This may differ slightly from study\_date\_anon in the clinical data which represents the date the report was signed.|
|StudyDescription|The exam type - for example a screening or diagnostic mammogram. On occasion, the study type may differ than what was recorded in the clinical data sheet due to mistakes in data entry at the time of acquisition. Images and other fields can be reviewed to troubleshoot, or these exams can be discarded. |
|SeriesDescription|The name of the series for an exam. This can vary depending on 2D, 3D, or C-view images but typically contains the type of view (CC, MLO, etc) and/or the laterality. This field is not frequently required. See *FinalImageType* and *ImageLateralityFinal*|
|FinalImageType|<p>Derived by combining information from several other DICOM fields to ascertain the image type</p><p>**2D**: standard 2D digital mammogram</p><p>**3D**: digital breast tomosynthesis (DBT). These cannot be currently used as they are locked in a proprietary container and *are not part of the AWS Open Data release.*</p><p>**C-view**: synthetic 2D image derived from the DBT</p><p>**ROI\_SSC/ROI\_SS**: screensave images annotated by the radiologist with a circlular ROI burned into the pixel data. These ROIs have already been extracted and mapped to their source image. *These are not part of the AWS Open Data release.*</p>|
|ImageLateralityFinal|<p>Derived by combining information from several other DICOM fields.</p><p>**L**: left breast</p><p>**R**: right breast </p>|
|ViewPosition|Type of view acquired such as CC or MLO|
|spot\_mag|<p>Indicates if the image is a special view such as spot compression or magnificiation.</p><p>**0**: image is a full field digital mammogram (FFDM). All screening studies are FFDM.</p><p>**1:** image is a special. Often used in diagnostic exams but should not occur for screening exams.</p>|
|num\_roi|Number of ROIs for a given file|
|<p>ROI\_coords</p><p></p><p></p>|<p>Coordinate(s) of any detected ROI on the image, represented as a list of lists. Sublists contains corner coordinates for ROI in the format ‘ymin, xmin, ymax, xmax’.</p><p>For 2D and C-view images, this field is the location of the ROI on the image. </p><p>For the  screensave (ROI\_SSC/ROI\_SS) images, this field is the location of the burned in ROI on the screensave image which serves as the source of the ROI location. *Screensaves are not part of the AWS Open Data release.*</p>|
|<p>SRC\_DST</p><p></p>|<p>A list of the source/destination file of each ROI in the ROI\_coords to be used for troubleshooting in case of suspicion that an ROI is applied to/from the wrong image.</p><p>For rows where the image is a 2D or C-view, this field contains the source file of the screensave image that ROI was extracted from. </p><p>For a row where the image is a screensave, this field contains the 2D or C-view image that the ROI was copied to. </p><p></p>|
|match\_level|<p>ROIs can be marked by radiologists on either 2D or C-view images at the time of interpretation. Because 2D and C-view images are very similar, ROIs that are applied to a 2D image can be translated to a C-view image and vice versa. *This is represented as a vector matching the length of ROI\_coords for a given image.*</p><p>**1**: indicates this ROI from the screensave was directly mapped to this image as a **primary match.** These ROIs are most reliable and have little to no errors.</p><p>**2:** indicates this ROI was generated as a **secondary** **match**. For example, if a screensave had a primary match to a 2D L CC view, the secondary match will be the C-view L CC view. These ROIs are slightly less robust and can be eliminated if you are experiencing noisy data.</p>|

**VIII. Special Note: Merging Clinical and Metadata files** 

A recurring challenge for many new users is merging information from the clinical and metadata files due to differences in the way these data are indexed. The clinical data is indexed by individual findings which can correspond to either breast and also varies by exam. The metadata is indexed by file wherein each row corresponds to one image. Both of these files can be linked by patient ID (empi\_anon), exam ID (acc\_anon), and laterality of the clinical finding and image.

If clinical and metadata are merged directly on the exam ID alone (acc\_anon), findings that relate to one breast in the  clinical data will be mapped to images from both breasts in the metadata. Instead, we must match clinical findings of the left breast with files of the left breast, clinical findings of the right breast to files of the right breast, and clinical findings of both breasts to all files.

To do this, use the *side* field in clinical data (L, R, B) and only merge rows where *side* matches the *ImageLateralityFinal* field in metadata.

- clinical data -  *side* L -> metadata - *ImageLateralityFinal* L
- clinical data  - *side* R -> metadata - *ImageLateralityFinal* R
- clinical data - *side* B -> metadata - all files (L and R)
- clinical data - *side* NaN -> metadata - all files (L and R)

This resultant table will have each clinical finding mapped to its corresponding files in metadata. Note that files **can** still be repeated in the joined table. For example, if a given accession has 2 findings in the left breast, each finding will be mapped to all the left breast images, resulting in each left breast image appearing twice in the resultant dataframe - once for *numfind1* and again for *numfind* 2. This is expected behavior.

**IX. Special Note: Pathology Results in Clinical Data**

In typical mammography screening workflow, an abnormal screening study (BIRADS 0) proceeds to a diagnostic study. If the diagnostic study is deemed suspicious (BIRADS 4 or 5), the patient will proceed to biopsy followed by surgery if indicated.

Each of these pathology results is recorded by a human administrator and linked the exam and finding from which they originated. That means that if a screening study has a finding that triggers diagnostic exam and biopsy, the resultant pathology result would be recorded on rows for **both** the screening and diagnostic exams in the clinical dataframe. Similarly, if the patient proceeds to surgery, the pathology result from surgery will also be recorded in the appropriate rows for both exams. **However, because this is a manual process there is potential for error in which the pathology result is recorded only to the diagnostic study and not the screening exam.** In this case, it may be useful to create a table of patient ID, pathology specimen date, and pathology results and then manually attribute results back to prior studies based on a date range.
