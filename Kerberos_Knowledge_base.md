# Kerberos Issue Resolution

### Problem Statement:
If logs indicate a failure due to a Kerberos issue and the client cannot authenticate using `[TOKEN, KERBEROS]`, take the following actions.
But to view the logs use the following information

- **Job Name**: `$Job_Name` (Get the job name from Incident Description)
- **Description**: Runs a Hadoop script to migrate the files.
- **Log Directory**: `"D:\Cognizant\Logs\Autosys_Logs.txt"`

Sample Incident Description Format/Template
>"myTechExpress ASSGNED:SEV4 Incident INC0000111##### $JobName OPSREPORTING"

### Action 1: Force Start the Job
Initiate a ServiceNow request via myTechExpress (mte) to force start the job. Use the following JSON template:

```json
{
    "Service Catalog": "Autosys Temporary Services",
    "AA Code": "6abc",
    "Job Type": "Force Start a Box Job",
    "Job Name": "$Job_Name",
    "Description": "Force starting due to Kerberos Issue"
}
```

### Action 2: Notify Users
Inform users that the job has been force started and ask them to monitor the job.