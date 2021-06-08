

# Load the data
heart_disease <- read.csv("C:\\Users\\Utilisateur\\Downloads\\Clustering Heart Disease Patient Data\\datasets\\heart_disease_patients.csv")

# Evidence that the data should be scaled?
summary(heart_disease)

# Remove id
heart_disease$id <- NULL

# Scaling data and saving as a data frame

scaled <-scale(heart_disease)

# What does the data look like now?
summary(scaled)

# Set the seed so that results are reproducible
seed_val  <- 10
set.seed(seed_val)

# Select a number of clusters
k=5

# Run the k-means algorithm
first_clust = kmeans(scaled,k,nstart=1)

# How many patients are in each cluster?
patient_in_cluster <- data.frame(cluster=c(1:k),n_patients=first_clust$size)

ggplot(patient_in_cluster, aes(x=cluster,y=n_patients)) + geom_bar(stat = "identity")



# Set the seed
seed_val <- 38
set.seed(seed_val)

# Select a number of clusters and run the k-means algorithm
k=5
second_clust = kmeans(scaled,k,nstart=1)

# How many patients are in each cluster?
patient_in_cluster <- data.frame(cluster=c(1:k),n_patients=second_clust$size)

ggplot(patient_in_cluster, aes(x=cluster,y=n_patients)) + geom_bar(stat = "identity")



# Add cluster assignments to the data
heart_disease["first_clust"] <- first_clust$cluster
heart_disease["second_clust"] <- second_clust$cluster

# Load ggplot2
library(ggplot2)

# Create and print the plot of age and chol for the first clustering algorithm

plot_one  <- ggplot(heart_disease, aes(x=age,y=chol,color=as.factor(first_clust))) + geom_point()

plot_one 

# Create and print the plot of age and chol for the second clustering algorithm
cluster2=as.factor(second_clust)
plot_two  <- ggplot(heart_disease, aes(x=age,y=chol,color=cluster2)) + geom_point()

plot_two

# Execute hierarchical clustering with complete linkage
hier_clust_1 <- hclust(dist(scaled), method = "complete")

# Print the dendrogram
plot(hier_clust_1)

# Get cluster assignments based on number of selected clusters
hc_1_assign=cutree(hier_clust_1,k=5)

# Execute hierarchical clustering with single linkage
hier_clust_2 <- hclust(dist(scaled), method = "single")

# Print the dendrogram
plot(hier_clust_2)

# Get cluster assignments based on number of selected clusters
hc_2_assign=cutree(hier_clust_2,k=5)

# Add assignment of chosen hierarchical linkage
heart_disease["hc_clust"] <- hc_1_assign

# Remove the sex, first_clust, and second_clust variables
hd_simple = heart_disease[, !(names(heart_disease) %in% c('sex', 'first_clust', 'second_clust'))]

# Get the mean and standard deviation summary statistics
clust_summary <- do.call(data.frame, aggregate(. ~ hc_clust, data = hd_simple, function(x) c(avg = mean(x), sd = sd(x))))
clust_summary

# Plot age and chol
plot_one  <- ggplot(hd_simple, aes(age,chol)) + geom_point(aes(color = as.factor(hc_clust)))
plot_one 

# Plot oldpeak and trestbps
plot_two  <- ggplot(hd_simple, aes(oldpeak,trestbps)) + geom_point(aes(color = as.factor(hc_clust)))
plot_two