# Bibliography summary

### Mutual information

#### Wiki 
For two variables $X,Y$ the mutual informaiton can be expressed as a function of entropy:

$I(X,Y) = H(X) + H(Y) - H(X,Y)$

Given a discrete random variable $X$, with possible outcomes $x_1, ..., x_n$ which  occur with probabilities $P( x_k )$  the entropy of $X$ is formally defined as: $H(X) = - \sum_{i=1}^n P(x_k) log(P(x_k))$.

#### Python implementation 

In the case of time series, we can seperate the different values taken by the coefficients by discrete bins. 

    histo, bins = np.hist(a_n, bins=50)

This function will create a histogramm of the coefficient $a_n(t)$ for 50 bins. The probavility of a value being in a bin is then simply:

    p = histo/np.sum(histo)

With these probabilities we can compute the Shannon Entropy with the above formula, which then enable us to compute the mutual information.

    mutual_info = MutualInfo(a, b, bins)

This is then computed for all the pairwise elements of the data.

### Transformers

Teacher forcing pros: it corrects the model from going to far off track. Cons: the training and prediction become two different exercices. A solution could be to have some teacher forcing at first during training and the more the epochs advance the less use teacher forcing. 

