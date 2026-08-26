
Part 1: Warmup — Cloud Concepts
======================================================================================================

1. Cloud Concepts Question 1
============================

What is the core economic model of cloud computing, and how does it differ from owning your own servers?
--------->

The core economic model of cloud computing is “pay-as-you-go”: We can rent the computing resources such as storage, processing power, and databases from a cloud provider and pay based on what is used. We dont need to buy our own resources and manitain them.

Cloud computing vs. owning servers
------------------------------------
Cloud: Low upfront cost. We pay for resources as we use them and can easily scale up or down.
Own servers: High upfront cost. We have to purchase hardware, maintain it, provide electricity and cooling, and replace or upgrade it over time.

Cloud: The cloud provider handles much of the infrastructure maintenance.
Own servers: The organization is responsible for maintenance, repairs, security, and capacity planning.
========================================================================================================

2.Cloud Concepts Question 2
============================
What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.
------------>

Vertical scaling means making one machine more powerful by adding resources such as CPU, RAM, or a faster GPU. For example, we might upgrade a machine with more RAM when a model-training job needs more memory.

Horizontal scaling means adding more machines and distributing the workload among them. For example, a web application might add more servers when traffic suddenly increases.

A web app that goes from 1,000 to 100,000 users would use horizontal scaling because it can add more machines to handle the increased traffic.

A model-training job that needs a faster GPU and more RAM would use vertical scaling because it needs a more powerful machine.

A pipeline that grows from 10 to 10,000 files would use horizontal scaling because the work can be divided among multiple machines.

========================================================================================================

Cloud Concepts Question 3
===========================

Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.
Gmail
Azure Virtual Machines
AWS S3 (Simple Storage Service)
GitHub Codespaces
Snowflake
Supabase
Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.
------------------->
Item	        Classification	    Reason
-------------------------------------------
Gmail	        SaaS	            You use the application without managing the servers or underlying 
                                    infrastructure.

Azure Virtual   IaaS	            You rent virtual machines but are responsible for the 
Machines                            operating system and software you install.

AWS S3	        IaaS	            It provides basic cloud infrastructure for storing 
                                    data without being a complete application.

GitHub 	        PaaS	            It provides a managed development environment where you can write  Codespaces                          and run your code without managing the underlying machines.

Snowflake	    PaaS/managed data   It provides managed data storage and analytics capabilities without 
                platform            requiring you to manage the underlying cloud infrastructure.

Supabase	    BaaS	            It provides application-level services such as a database, 
                                    authentication, and storage through APIs.

IaaS: IaaS provides basic computing infrastructure such as virtual machines, storage, and networking. For example, Azure Virtual Machines lets me rent a machine, but I am responsible for the operating system, software, configuration, and security updates.

PaaS: PaaS provides a managed platform where I can deploy my code without managing the underlying infrastructure. For example, with GitHub Codespaces, the environment is provided for me, so I mainly manage my code and development setup.

SaaS: SaaS is a complete application that I use without managing the infrastructure behind it. For example, with Gmail, I simply use the email application while the provider manages the servers, software, and maintenance.

========================================================================================================

Cloud Concepts Question 4
==========================
What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?
------------->

A managed data platform such as Databricks or Snowflake provides preconfigured tools for data processing, analytics, and machine learning on top of cloud infrastructure. Compared with using AWS or GCP directly, we gain simplicity and faster setup, but we give up some flexibility and may pay more for the convenience.

=======================================================================================================

Cloud Concepts Question 5
==========================
The lesson names two situations where the cloud is probably not the right choice. What are they?
-------------->

The cloud is probably not the right choice when the dataset fits comfortably on one local machine and there is no large computing requirement, because local processing may be faster and cheaper. It may also not be the right choice when the learning and setup complexity of cloud infrastructure outweighs the benefits for a simple project.

======================================================================================================

Part 2: Warmup — Cloud Landscape
=======================================================================================================

Cloud Landscape Question 1
---------------------------
Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.
-------------->

AWS: AWS is the largest and broadest cloud provider, making it common in enterprises, startups, and organizations that need many different cloud services.

GCP: GCP is particularly strong in data analytics and machine learning, so it is often used by organizations doing large-scale data or ML work.

Azure: Azure is especially strong in enterprise and government environments because of its integration with Microsoft products and services.

========================================================================================================

Cloud Landscape Question 2
--------------------------
The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.
Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?
--------------->

The course switched to Supabase because students can create an account themselves quickly, whereas Azure can require organizational invitations and configuration.

Supabase uses a relational database with tables and SQL, which makes it easier for students to learn transferable data skills than Azure Blob Storage.

Supabase fits the course's ETL pipeline well because the raw and enriched data can be stored in related tables that are easy to inspect and query.

My reflection: This suggests that when starting a new project, I should evaluate a cloud tool based not only on its features but also on its accessibility, learning curve, cost, and how well it fits the project's actual needs.

========================================================================================================

Cloud Landscape Question 3
---------------------------
For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

You need to store 10 TB of image files and retrieve them by filename from any machine.
You need to run an ML training job on a GPU for four hours, then shut it down.
You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
You need to send structured data to a large language model and get a text response back.
------------------->
Object storage — AWS S3: I could use S3 to store 10 TB of image files and retrieve them by their filenames or object keys.
Compute — AWS EC2: I could use an EC2 instance with a GPU to run the ML training job for four hours and then shut it down.
Serverless compute — AWS Lambda: I could use Lambda to host functions that automatically scale based on demand.
LLM API — Azure OpenAI: I could send structured data to an LLM through Azure OpenAI and receive a text response.

========================================================================================================
Cloud Landscape Question 4
---------------------------
The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?
--------------->

I could build a weather-analysis application that stores raw weather data in Supabase and uses Google BigQuery to perform large-scale analytics. I could also use an LLM API to generate summaries of the weather data.

Using one provider can simplify management, billing, security, and integration between services. However, I would give up some flexibility and might lose access to specialized services that another provider offers.

========================================================================================================




