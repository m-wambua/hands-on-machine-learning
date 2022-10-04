# Machine laerning is  great for:
#Problems for solutions that require a lot of hand tuning or long lists
# of rules: one machine learning can often simplify code and perform better
# Complex problems for which there is no good solution at all using traditional approach:
# the best machine learing techniques can find a solution
# Flactuating environments: a mchine laerning system can adapt to new data.
# Getting insights about complex problems and large amounts of data.#
""" CHAPTER 1
Types of machine learning Systems:

1.Supervised/unsupervised  learning- there are 4 major categories.
    a.supervised-the training data you feed to the algorithm includes the desired solutions called labels.a typical learning
    task is classification.(the spam filter). Another typical task is to predict a target numeric value, such as the orice of a 
    car given a set of features called predictors. this sort of task is called regression.
    Important supervised machine learning algorithms include:
        k-nearest neighbors
        linear regression
        logistic regression
        support vector machines(SVMs)
        Decision Trees and Random Forests
        Neural networks
    b.unsupervised-the training data is unlabeled. some important algorithms are:
        clustering
            K-Means
            DBSCAN
            Hierachical Cluster Analysis(HCA)
        anomaly detection and novelty detection
            one class SVM
            isolation Forest

        Visualization and dimensionality reduction

            principal component analysis(PCA)
            kernel PCA
            locally-linear embedding(LLE)
            t-distributed Stochastic neighbor embedding(t-SNE)
        Association rule learning
            Apriori
            Eclat
    c.semisupervised
    d.reinforcement learning
 as an example, say you have a lot of data about your blog's visitors.You want to run a clustering algorithm 
 to detect groups of similar visitors.At no point do you tell the algorithm which group a visitor belongs to:
 it finds those connections without your help.For example,it might notice that 40% of your visitors are males
 who love comic books and generally read your blog in the eveining, while 20% are young sci-fi lovers
 who visit during the weekends, and so on.If you use a hierarchical clustering algorithm, it may also subdivide each group into
 smaller groups.This may help you target your posts for each group.Visualition algorithms are also good examples of
 unsupervised learning algorthms: you feed them a lot of complex and unlabled data, and they output a 2D or 3D representation 
 of your data that can easily be plotted.A related task is dimensionality reduction, in which the goal is to simplify the data
 without loosing too much information.One way to do this is to merge several correlated features into one.For example a cars 
 mileage may be correlated to its age, so the dimensionality reduction algorithm will merge them into one feature that
 represnts thr car's wear and tear.This is called feature extraction.Yet another important unsupervised task is anomaly detection
 for example,detection unusual credit card transactions to prevent fraud,catching manufacturing defects, ot automatically 
 removing outliers froma dataset before feeeding it another learning algorithm.The system is shown mostly normal instances during
 training, so it learns to recognize them and when it sees a new instance it can tell whether it looks like a normal one whether
 it is likely an anomaly.A very similar task is novelty detetion:the difference is that novelty detection algorithms expect to see
 only normal data during training,while anomaly detection algorithms are usually more tolerant,they can often perform well
 even with a small percentage of outliers in a training set.
 Finally,another common unsupervised task is association rule learning,in which the goal is to dig into large amounts of data and
 discover interesting relations between an attributes.For example,supposed you own supermarket.Running an association rule
 on your sales logs may reveal that people who purchae barbecue sauce and potato chips also tend to buy steak.Thus,you may want 
 to place these items close to each other.

 Semisupervised learning
    Some algorithms can deal with partially labeled training data,usually a lot of unlabeled data and a little bit laballed data.
    Some photo-hosting service ,such as Google Photos, are good example of this .Once you upload all your family photos to the 
    service,it automatically recognizes that the same person .This is the unsupervised part of the algorithm(clustering).Now 
    all the system needs is for you to tell it who these people are.Just one label per person,and it is able to name everyone 
    in everyone in every photo,which is useful for searching photos.Most semisupervised learnig algorithms are combinations 
    of unsupervised and supervised algorithms.For example Deep Belief Networks(DBNs) are based on unsupervised components called
    Restricetd Boltzmann Machines(RBMs) stacked on top of another.RBMs are trained sequentially in an unsupervised manner,and
    then the whole system is fined-tuned using supervise learning techniques.

Reinforcement learning
    The learinig systems,called an agent in this context,can observe the environment,select and perform actions,and get rewards
    in return (or penalties in the form of negative rewards).It must  then learn by itself what is the best strategy,called a 
    policy,to get the most reward over time.A policy defines what action the agent should choose when it is in a given situation.


Batch and Online Learning
    In batch learning the system us incapable of learning incrementally:it must be trianed usually using all the availale data.
    This will usually take a lot of time  computing resources, so it is typically done offline.Fisrt the system is trained, and 
    then it i launched into production and runs without learning anymore; it just applies what it has learned.This is called 
    offline learning.If you want  a batch learning system to know about new data,you need to train a new version of the system
    from scratch on the full dataset then stop the old system and replace it with a new system.

Online learning 
    You train the data incrementally by feefing it data instancess sequentially,either individually or by small groups 
    called mini batches.Each learning step us fast and cheap ,so the system can learn about new data on th  fly as it 
    arrives.Online learning is greate for systems that recieve data as a continuous flow and need to adapt rapidly or 
    autonomously.It is aslo a good option if you have limited computing resources.Online learning algorithms can aslo
    be used to train on huge datasets that cannot fit in ones's machine's main memory (out-of-core learning).One
    important thing about online learning systems is how fast they should adapt to changing data: this is called the 
    learning rate.

Instance-Based Versus Model-Based Learning
    Another way of categorizing machine learning algorithm is how they generalize.Most machine learning tasks are about 
    making predictions.This means that given a number of training examples ,the system needs to be able to generalize to examples
    it has ever  seen before.There are two main approaches to generalization:instance-based learning and model-based learning.
Instance-based learning
    Learns the examples by heart, then generalizes to new cases by comparing them to the learned examples(or a subset of them)
    ,using a similarity measure.
Model-based learning
    Another way to generalize from a set of examples us to build a model of these examples , then use that model to make 
    predictions


In summary for modelling:
    You studied the data.
    You selected a model
    You trained it on the training data(i.e the learning algorithm searched for the model 
    parameter values that minimize a cost function)
    finally , you applied the model to make the predictions on new cases(this is called 
    inferene)

So what can go wrong in learning and prevent you from making accurate predictions.

MAIN CHALLENGES OF MACHINE LAERNING
    Insufficient Quantity of training data
        It takes a lot of data for most machine learning algorithms to work 
        properly.

    Nonrepresentative training data-in order to generalize well, it is crucial
    that your training data  be representative of the new cases you want to generalize

Poor quality data-if your training data is full of errors ,outliers and noise , 
it will make it harder for the system to detect the underlying patterns,so your 
system is less  likely to perform well.It is often well worth the effort to spend
time cleaning up your training data

Irrelevant features-your system will only be capable of learning if the training 
data contains enough relevant features and not too many irrelevant ones.A good 
part of the success of a machine learning project is coming up with a good set of 
features to train on.This process is called feature engineering and involves:
    feature selection - selecting the most useful features to train on among the 
    existing features

    feature extraction-combining exixting features to train on among existing 
    features

Overfitting the training data-this means the  model performs well on the training 
data, but it doesn't generalize well.Overfitting happens when the model us too
complex relative to the amount and noisiness of the training data.The possible 
solutions are:
    To simplify the model by selecting one with fewer parameters by reducing the 
    number of attributes in the training data or by constraining the model
    To gather more training data
    To reduce the noise in the training data.
Constraining a model to make it simplier and reduce the risk of overfitting is called
regularization.

Underfitting the training data-it occurs when the model is too simple to learn 
the underlying structure of the data.Ways of ficing  this problem include:
    selecting a more powerfyl model with more parameters
    Feeding better features to the learning algorithm
    Reducing the contraints on the model.

Testing and Validating

The only way to know how well a model will generalize to new cases is to 
actually try out new cases .One way to do that is to put your model un production
and monitor how well it performs.A good way to do this is to split  your data into two:
    training set
    test set
The error rate on new cases is called the generalization error and by evaluating
your model on the test set ,you get an estimate of this error.This value tells 
how well your model will perform on instances it has never seen before.If the 
training error is low(i.e your model makes few mistakes on the trainig=ng set) but
the generalization error is high,it means that your model us overfitting the training
data.
 
Hyper parameter tuning and Model selection
Hold out vaalidation-you simply hold out part of the training set to evaluate 
several candidate models and select the best one.More specifically you train
 multiple models with various hyperparamters on the reduced training set and 
 you select the model that performs best on validation set.After this holdout 
 validation process, ypu train the best model on the full training set and this 
 gives you the final model.Lastly, you evaluate this final model on the test 
 set to get an estimate of the generalization error.



 Data mismatch


"""

""" CHAOTER 2

Frame the problem

Pipelines-a sequence of data processing components is called a data pipeline

selecting a performance measure
a typical performance measure for regression problems is the root mean square 
error(RMSE). It gives  an idea of how much error the system typically makes in its
predictions,with a higher weight for large errors.
Others include the :
    Mean Absolute Error -it is measuresd by taking the average
    of the absolute difference between 
    actual values and the predictions.

    Coefficint of Determination or R^2 - it measures how
    well the actual outcomes are replicated by the 
    regression line.It helps you to understand how
    well the independent variable adjusted
    with variance in your model.

    Adjusted R-squared-the value of
    R^2 keeps on increasing with the additcion of
    more independent variables even though
    they may not have a significant 
    impact on the prediction.

Check the Assumption - it is good practice to list and verify the assumptions that
were made so far; this can catch serious issues early on
"""