# Incident Reference for MDM Queries

This manual outlines the assortment of incidents encountered by the Master Data Management (MDM) unit, which are reported via the ServiceNow framework.

The incidents that are considered suitable for action include:
1. EBC_Incident
2. Monitoring_Alert
3. Autosys_Issue

## EBC_Incident

Routinely, the MDM division conducts synchronization of contact information where they incorporate data dispatched from the Customer Hub (CHUB) division. The database managed by the CHUB is the origination point for data that is then migrated into the MDM's own database. The EBC systems then retrieve this data through the MDM's API services. Issues have been reported where there are inaccuracies in the contact details tied to particular policies within the EBC system.

### Source Data Configuration

The account contact information is sourced from three distinct SQLite Tables:
- **ENT_EMAIL_DETAILS**: This table archives email contact info for each enterprise, with the ENT_ACCT_ID column serving as the enterprise identifier.
- **ENT_POL_DETAILS**: This table catalogs information concerning the policies held by each enterprise, given that enterprises may hold multiple policies.

To pull data from the ENT_EMAIL_DETAILS table, the query prioritizes records by the following SQL window function:
   
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
Within this schema, CIAM-derived contacts are considered more authoritative over those from PC.

Incident Case Study:
An incident was reported where the contact details for policy number 08SBAAA8624 were not presented correctly in the EBC system. The expected contact details for the insured should have been "John Buckley," yet "CHARLES CARROLL" is what appears. Upon review, it has been found that the MDM service is supplying incorrect information. Documented evidence such as screenshots and API request-response pairs have been submitted for review. This has prompted a call to resolve the issue.

For similar incidents regarding contact information discrepancies, classify the incident under:

> "Incident_Type" = "EBC_Incident"

## Monitoring_Alert

Instances of this incident are triggered when a monitored API's response duration exceeds 3 seconds, sparking the creation of an alert. Incidents of this type are typically self-resolving and may not require intervention.

Example Incident:
>"Critical alert [Alert4760849] initiated for CI: [Commercial Account Management 1.0-PROD]. The metric is [] indicating [Host or monitoring unavailable], derived from data via [DYNATRACE].  
The total number of services affected under CI: [Commercial Account Management 1.0 -PROD] are 2"

Such incidents, as exemplified above, should be categorized as:

> "Incident_Type" = "Monitoring_Alert"

These incidents are representative of monitoring alerts.

## Autosys_Issue

The majority of incidents for Autosys issues are due to Kerberos problems in the Hadoop environment. These issues typically arise when multiple jobs attempt to access network authentication simultaneously.

The job in question is primarily used to ingest data from the source and place it into the data lake. If the job fails and an incident is created, you should view the logs to diagnose the problem.

Example of Autosys_Issue (Extract job name from the actual incident description input):
> "myTechExpress ASSGNED:SEV4 Incident INC000011119999 6abcp5cmd_ag_log OPSREPORTING"

Such incidents, as exemplified above, should be categorized as:
> "Incident_Type" = "Autosys_Issue"