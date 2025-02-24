## Master Data Management (MDM) Synchronization Issue

The Master Data Management (MDM) team updates contact details using information from the Customer Hub (CHUB). This involves transferring data from the CHUB's primary database to the MDM's database. The Enterprise Business Console (EBC) systems access this updated contact information via MDM's API services. However, there are instances where the EBC system shows incorrect contact details for certain policies. To validate this mismatch do an analysis on the CHUB Source database using the following information.

### Data Collection Process

Data is fetched from these CHUB SQLite database tables:
- **ENT_EMAIL_DETAILS**: Stores email contact details. The ENT_ACCT_ID column is the unique identifier for each enterprise account.
- **ENT_POL_DETAILS**: Records policy-related data for enterprises, ensuring a comprehensive list of policies an enterprise might have.

Data extraction involves the following SQL query on the ENT_EMAIL_DETAILS table. The query ranks records based on the source system (SRCE_SYS_CD), prioritizing the systems as follows: CIAM, PC, CLDW, PARIS. The most recent update (TRANS_ACTV_TMSP) is also considered:

This Data extraction is used for CHUB>MDM. Not for MDM to EBC.

```sql
ROW_NUMBER ()
       OVER (
           PARTITION BY ENT_ACCT_ID
           ORDER BY
               (CASE
                    WHEN SRCE_SYS_CD = 'CIAM' THEN 1
                    WHEN SRCE_SYS_CD = 'PC' THEN 2
                    WHEN SRCE_SYS_CD = 'CLDW' THEN 3
                    WHEN SRCE_SYS_CD = 'PARIS' THEN 4
                END) ASC,
               TRANS_ACTV_TMSP DESC)
           AS RNUM
```

CIAM system contacts are given the highest priority, followed by PC, CLDW, and PARIS. Use this ranking logic to investigate and understand any data discrepancies. Based on your findings, compile a report on the email mismatches in EBC Systems based upon the fetching details from CHUB Source to MDM Database.

### Detailed Discrepancy Findings

- For `Policy_ID`: `##ABCDE####`, the actual email is the same as the expected, indicating no discrepancy.
- For `Policy_ID`: `##ABCDE####`, the actual email differs from the expected. Upon investigation, it was discovered that the `Expected_Email_ID` with SRCE_SYS_CD 'CIAM' had an older timestamp compared to the `Actual_Email_ID`, which was submitted later from the system 'PC'. The discrepancy occurred due to the priority condition being correctly applied but the newer submission time of the 'PC' source entry taking precedence.

In conclusion, the analysis revealed that expected email addresses are not showing up in certain instances due to more recent contact updates from different source systems taking precedence over older records in the priority condition as per the defined MDM hierarchy. Each case needs to be investigated individually to ensure alignment with current data governance policies and possibly adjust the system to accommodate recent changes that may be legitimately replacing older contact data.