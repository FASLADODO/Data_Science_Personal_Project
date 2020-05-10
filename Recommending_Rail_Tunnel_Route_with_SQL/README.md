# Recommending Rail Tunnel Route with SQL

## Objective
The purpose of this project is to use SQL statement in order to find the best route
for a fictional transportation company that wants to build a rail tunnel that connects two airports in the US based on several requirements or criteria. The data being used
for this project are saved in Hadoop Distributed File System (HDFS) and AWS S3. In order to query the data, SQL Engine such as Apache Hive or Impala will be used.

## Files
There are seven files and one folder in this project. These files are:

- SQL_Project.pdf - This is the report of this project, which explains the step-by-step process and the detailed requirements for SQL statement.
- fly_DB_overview.pdf - The overview of the database in HDFS where all the data regarding US flights and airports are stored.
- query_head_hourly_central.xlsx - First five rows of the query result regarding tunnel boring machine (TBM) #1, which is stored in AWS S3.
- query_head_hourly_north.xlsx - First five rows of the query result regarding tunnel boring machine (TBM) #2, which is stored in AWS S3.
- query_head_hourly_south.xlsx - First five rows of the query result regarding tunnel boring machine (TBM) #3, which is stored in AWS S3.
- query_tbm_sf_la.xlsx - First five rows of the union of all of the TBM data (1-3).
- tunnel_recommendation_query_result.xlsx - Query results of the tunnel route recommendation to connect two US airports.
