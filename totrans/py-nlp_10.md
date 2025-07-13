
# Advanced Tools

This appendix focuses on how the various frameworks can be used in NLP applications. We will look at an overview of the frameworks and touch on the basic features and what they do for you. We are not going to see look at a detailed architecture of each framework. Here, the purpose is to get you aware of the different tools and frameworks that can be used together to build various NLP applications. We will also look at visualization libraries that can help you develop a dashboard.

# Apache Hadoop as a storage framework

Apache Hadoop is one of the widely used frameworks. Hadoop allows the distributed processing of large datasets across clusters of commodity computers using a simple programming model. Hadoop uses the concept of MapReduce. MapReduce divides the input query into small parts and processes them in parallel to the data stored on the **Hadoop distributed file system** (**HDFS**).

Hadoop has the following features:

*   It is scalable
*   It is cost-effective
*   It provides a robust ecosystem
*   It provides faster data processing

Hadoop can be used as a storage framework for NLP applications. If you want to store large amounts of data, then you can use a multinode Hadoop cluster and store data on HDFS. So, many NLP applications use HDFS for their historical data. Hadoop sends a program to the data and the data processes it locally. These features give Hadoop good speed. Note that Hadoop performs operations on the disk level, which is slow, but we execute operations in parallel so data processing is fast. Now, you may think that disk operations are slow compared to memory operations, but we have large amounts of data, which will not fit into memory at once. So, this approach of processing data locally and executing operations in parallel, using a multinode cluster, gives us a good throughput.

Hadoop has the following components as part of its core architecture:

*   HDFS
*   MapReduce
*   YARN
*   Hadoop common utilities

You can see the architecture of Hadoop in *Figure 01*:

![](img/3a83ddd1-6c61-4781-9061-b9f04f5667cf.png)

Figure 01: Hadoop 2.x yarn architecture(Image credit: https://github.com/zubayr/big_config/blob/master/hbase/hbase_tuning.md)

You can see the Hadoop ecosystem in *Figure 02*:

![](img/a9cef694-fe42-41d9-91e1-e65a1d3bf4bf.png)

Figure 02: The Hadoop ecosystem (Image credit: https://s3.amazonaws.com/files.dezyre.com/images/blog/Big+Data+and+Hadoop+Training+Hadoop+Components+and+Architecture_1.png)

For real-time data processing, Hadoop is a bit slow and not very efficient. Don't worry! We have another framework that helps us with real-time data processing.

Many NLP applications use Hadoop for data storage because it can handle data processing very well. On a personal level, I used Hadoop to store my corpus on HDFS. Then, I have used Spark MLlib to develop **machine learning** (**ML**) algorithms. For real-time data processing, I use Apache Flink.

For experimenting purposes, I have provided you with the steps of setting up a single-node Hadoop cluster. The GitHub link for this is: [https://github.com/jalajthanaki/NLPython/blob/master/Appendix3/Installationdocs/App3_3_Hadoop_installation.md](https://github.com/jalajthanaki/NLPython/blob/master/Appendix3/Installationdocs/App3_3_Hadoop_installation.md).

You can find some of the commands of Hadoop in this document:

*   [https://dzone.com/articles/top-10-hadoop-shell-commands](https://dzone.com/articles/top-10-hadoop-shell-commands).
*   [https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/FileSystemShell.html](https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/FileSystemShell.html)

# Apache Spark as a processing framework

Apache Spark is a large-scale data processing framework. It is a fast and general-purpose engine. It is one of the fastest processing frameworks. Spark can perform in-memory data processing, as well as on-disk data processing.

Spark's important features are as follows:

*   **Speed**: Apache Spark can run programs up to 100 times faster than Hadoop MapReduce in-memory or 10 times faster on-disk
*   **Ease of use**: There are various APIs available for Scala, Java, Spark, and R to develop your application
*   **Generality**: Spark provides features of Combine SQL, streaming, and complex analytics
*   **Run everywhere**: Spark can run on Hadoop, Mesos, standalone, or in the cloud. You can access diverse data sources by including HDFS, Cassandra, HBase, and S3

I have used Spark to train my models using MLlib. I have used Spark Java as well as PySpark API. The result is you can redirect to the HDFS. I have saved my trained models on HDFS and then loaded them as and when needed. Spark really speeds up your processing time. I have experienced this. The reason behind this is its in-memory processing architecture. Spark architecture is given in *Figure 03*:

![](img/22b44ae1-d342-45a5-8aed-40263890ab27.png)

Figure 03: Spark running architecture (Image credit: https://www.slideshare.net/datamantra/spark-architecture)

You can see the Spark ecosystem in *Figure 04*:

![](img/0efc3011-5cf0-4850-b821-cc786b50c45b.png)

Figure 04: Spark ecosystem (Image credit: http://jorditorres.org/spark-ecosystem/)

You can see the installation steps on this GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/Appendix3/Installationdocs/App3_4_Spark_installation.md](https://github.com/jalajthanaki/NLPython/blob/master/Appendix3/Installationdocs/App3_4_Spark_installation.md)

You can find more information on the following links:

*   [https://jaceklaskowski.gitbooks.io/mastering-apache-spark/content/](https://jaceklaskowski.gitbooks.io/mastering-apache-spark/content/)
*   [https://www.gitbook.com/book/jaceklaskowski/mastering-apache-spark/detail](https://www.gitbook.com/book/jaceklaskowski/mastering-apache-spark/details)
*   [http://spark.apache.org/](http://spark.apache.org/)
*   [http://spark.apache.org/docs/latest/ml-guide.html](http://spark.apache.org/docs/latest/ml-guide.html)
*   [http://spark.apache.org/docs/latest/mllib-guide.html](http://spark.apache.org/docs/latest/mllib-guide.html)

# Apache Flink as a real-time processing framework

Apache Flink is used for real-time streaming and batch processing. I have told you we should not worry about real-time frameworks. The reason is we have the Flink framework for this.

Flink is an open source stream processing framework for distributed, high-performing, always available, and accurate data streaming applications. You can see more about Flink at [https://flink.apache.org/](https://flink.apache.org/).

Flink will definitely provide a very nice future. You can see in *Figure 05*:

![](img/8b221644-574a-411c-b8d6-7cf90df17616.png)

Figure 05: Features of Flink (Image credit: https://flink.apache.org/)

Flink is quite a new framework. If you want to perform real-time sentiment analysis or make a real recommendation engine, then Flink is very useful. You can refer to the following video where you can understand how the HDFS, Flink, Kappa, and lamda architecture has been used. It's a must-see video:

[https://www.youtube.com/watch?v=mYGF4BUwtaw](https://www.youtube.com/watch?v=mYGF4BUwtaw)

This video helps you understand how various frameworks fuse together to develop a good real-time application.

# Visualization libraries in Python

Visualization is one of the important activities that is used to track certain processes and the results of your application. We used `matplotlib` in Chapter 6, *Advance Feature Engineering and NLP Algorithms*, as well as in other chapters.

Apart from `matplotlib`, we can use various visualization libraries:

*   `matplotlib`: It is simple to use and very useful
*   `bokeh`: It provides customized themes and charts
*   `pygal`: You can make cool graphs and charts with this

You can use the following links to refer to each of the libraries. All libraries have written documentation so you can check them and start making your own charts.

You can find more on `matplotlib` at [https://matplotlib.org/](https://matplotlib.org/).

You can find more on `Bokeh` at [http://bokeh.pydata.org/en/latest/docs/gallery.html](http://bokeh.pydata.org/en/latest/docs/gallery.html).

You can find documentation about `pygal` at [http://pygal.org/en/stable/documentation/index.html](http://pygal.org/en/stable/documentation/index.html).

# Summary

If you want detailed information regarding these frameworks and libraries, then you can use the Gitter room to connect with me, because in-depth details of the frameworks are out of the scope of this book.

This framework overview will help you figure out how various frameworks can be used in NLP applications. Hadoop is used for storage. Spark MLlib is used to develop machine learning models and store the trained models on HDFS. We can run the trained model as and when needed by loading it. Flink makes our lives easier when real-time analysis and data processing come into the picture. Real-time sentiment analysis, document classification, user recommendation engine, and so on are some of the real-time applications that you can build using Flink. The `matplotlib` is used while developing machine learning models. The `pygal` and `bokeh` are used to make nice dashboards for our end users.

