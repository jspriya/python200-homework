My supabase project is setup successfully.

Cloud Cost Analysis
-------------------

For Scenario A, the t3.micro EC2 instance running for approximately 160 hours per month costs $1.66 per month. This was not surprising because the instance is small and is only running for about 160 hours each month.

For Scenario B, the p3.2xlarge EC2 instance costs $2,233.80 per month, the RDS db.m5.large costs $323.03 per month, and 1 TB of S3 Standard storage costs $23.55 per month, for a total of $2,580.38 per month. I was surprised by how much more expensive the GPU-based EC2 instance was compared with the lightweight t3.micro instance. While exploring the calculator, I found it interesting to see how much the cost can increase when using more powerful compute resources and running them continuously.

The two scenarios have a very large cost difference: $1.66 versus $2,580.38 per month. This shows that GPU instances such as the p3.2xlarge are worth the cost when a workload genuinely needs GPU processing, such as machine learning or intensive analytics. For lightweight workloads that do not require a GPU, a smaller CPU-based instance can be much more cost-effective.


Project walk through youtube link  - https://youtu.be/mC5iEa0fz08
----------------------------------
