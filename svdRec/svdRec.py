# A module to handle recommendation systems based on sparse matrix decomposition.
# The basis is Singular Value Decomposition (SVD) and there are methods to 
# return similar items and find the N closest items for a user. 
#
# Written by Z. Miller 05/31/17. Open source if not used for profit.

import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from scipy.sparse.linalg import svds

class svdRec():
    def __init__(self):
        """
        The svdRec class does nothing upon spawn, save for create a few 
        variables that must exist for the class to work. The user must use
        a seperate call to load data in, since the class accepts multiple data
        types as input. 
        """
        self.U, self.s, self.V = (None,None,None)
        self.user_encoder = None
        self.item_encoder = None
        self.mat = None
        self.decomp = False
    
    def load_csv_sparse(self,filename, delimiter=',', skiprows=None):
        """
        Reads from a CSV file using numpy's 'loadtxt' function. The expected format for a row 
        is: 
        rowID, colID, Value, any, other, assorted, junk.
        This defines the base sparse matrix (self.mat), and converts it from the coordinate type matrix
        to a csr_matrix which is better for processing.
        
        Inputs: Filename (str) 
        Kwargs: delmiter for the csv, how many rows to skip (if there's a header)
        Returns: None
        """
        print("Note: load_csv_sparse expects a csv in the format of: rowID, colID, Value, ...")
        u, m, r = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows, usecols=(0,1,2)).T
        self.mat = coo_matrix((r, (u-1, m-1)), shape=(u.max(), m.max())).tocsr()
        print("Created matrix of shape: ",self.mat.shape)
        
    def load_data_numpy(self, array, data_type=float):
        """
        Converts a numpy matrix or array (both are valid inputs) directly into a csr
        sparse matrix for use with sparse SVD. This defines the base sparse matrix
        (self.mat) for later user. 

        Inputs: matrix (numpy array or numpy matrix) 
        Kwargs: data_type to force the sparse matrix into float/int/etc
        Returns: None
        """
        self.mat = csr_matrix(array,dtype=data_type)
        print("Created matrix of shape: ",self.mat.shape)

    def load_data_list(self, array, data_type=float):
        """
        Converts a list of lists into a csr sparse matrix for use with 
        sparse SVD. This defines the base sparse matrix (self.mat) for later usage. 

        Inputs: matrix (list of lists) 
        Kwargs: data_type to force the sparse matrix into float/int/etc
        Returns: None
        """
        self.mat = csr_matrix(np.array(array),dtype=data_type)
        print("Created matrix of shape: ",self.mat.shape)
        
    def load_item_encoder(self, d):
        """
        Takes in a dictionary of the format: {itemID: "item name"} to later be used
        to decode from ID space to named space if necessary. If never called,
        'get_item_name' returns a warning.

        Inputs: d (dictionary)
        Returns: None
        """
        if type(d) != dict:
            raise TypeError("Encoder must be dictionary with key = itemID and value = Title")
        self.item_encoder = d
        
    def load_user_encoder(self, d):
        """
        Takes in a dictionary of the format: {userID: "user name"} to later be used
        to decode from userID space to named space if necessary. If never called,
        'get_user_name' returns a warning.

        Inputs: d (dictionary) 
        Returns: None
        """
        if type(d) != dict:
            raise TypeError("Encoder must be dictionary with key = userID and value = Title")
        self.user_encoder = d
        
    def get_item_name(self,itemid):
        """ 
        Returns the name of an item given its ID number

        Input: Item id (int)
        Return: Name
        """
        if self.item_encoder:
            return self.item_encoder[str(itemid)]
        else:
            return "No ItemId -> Item-name Encoder Built!"
    
    def get_user_name(self,userid):
        """ 
        Returns the name of a user given it's ID number

        Input: User id (int)
        Return: Name
        """
        if self.item_encoder:
            return self.user_encoder[str(userid)]
        else:
            return "No UserID -> Username Encoder Built!"
    
    def SVD(self, num_dim=None):
        """
        Computes the matrix decomposition for the data, setting U, V, and s to the output matrices.
        This is based on 'svds', a singular value decomposition library designed to work with
        scipy sparse matrices.

        Input: num_dim (int): the number of dimensions for truncating SVD. 
        Returns: None
        """
        if num_dim==None:
            print("Number of SVD dimensions not requested, using %s dimensions." % (min(self.mat.shape)-1), "To set, use num_dim.")
            num_dim = min(self.mat.shape)-1
        self.U, self.s, self.V = svds(self.mat,k=num_dim)
        self.decomp = True
    
    def get_cell(self,i,j):
        """
        Return value for a cell in the sparse matrix.

        Inputs: row val (int), col val (int)
        Returns: value of cell
        """
        return self.mat[i,:].toarray()[0,j]

    def get_row(self,userID):
        """
        Helper function that returns raw data row for user.

        Inputs: userID (int) 
        Returns: User's Item ratings (numpy array)
        """
        return self.mat[userID,:].toarray()[0]
    
    def get_similar_items(self, itemID, num_recom=5, show_similarity=False):
        """
        Loops through all items and finds the overlap in the reduced vector space
        with the item input. Returns a list of the ID's for the most similar items, 
        as determined by the angular overlap. 

        Inputs: 
        item ID (int): The item to return similar items for, 
        num_recom (int): The number of items to recommend,
        show_similarity (bool): If True, the returned values are tuples (itemID, dot product)
        
        Returns:
        ItemID (int)
        Dot Product (float) [if show_similarity = True]
        """
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        recs = []
        for item in range(self.V.T.shape[0]):
                recs.append([item+1,self.item_similarity(itemID-1,item)])
        if show_similarity:
            final_rec = [(i[0],i[1]) for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        else:
            final_rec = [i[0] for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        return final_rec[:num_recom]
    
    def item_similarity(self,item1,item2):
        """
        Helper function to return item-item similarity via dot product.

        Input: item1 (int), ID of item1, item2 (int), ID of item2
        Returns: Dot product of item matrix elements (float)
        """
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.V.T[item1],self.V.T[item2])
    
    def user_similarity(self,user1,user2):
        """
        Helper function to return user-user similarity via dot product.

        Input: user1 (int), ID of user1, user2 (int), ID of user2
        Returns: Dot product of user matrix elements (float)
        """
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.U[user1],self.U[user2])
    
    def user_item_predict(self,user,item):
        """
        Helper function to return user-item similarity via dot product.

        Input: user (int), ID of user, item (int), ID of item
        Returns: Dot product of user-item matrix elements (float)
        """
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.U[user],self.V.T[item])
        
    def recommends_for_user(self, userID, num_recom=2, show_similarity=False):
        """
        Loops through all items and finds the overlap in the reduced vector space
        with the input user. Returns a list of the ID's for the most similar items, 
        as determined by the angular overlap. 

        Inputs: 
        user ID (int): The user to return recommended items for, 
        num_recom (int): The number of items to recommend,
        show_similarity (bool): If True, the returned values are tuples (itemID, dot product)
        
        Returns:
        ItemID (int)
        Dot Product (float) [if show_similarity = True]
        """
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        recs = []
        for item in range(self.V.T.shape[0]):
            recs.append((item+1,self.user_item_predict(userID-1,item)))
        if show_similarity:
            final_rec = [(i[0],i[1]) for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        else:
            final_rec = [i[0] for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        return final_rec[:num_recom]
    
    def recs_from_closest_user(self, userID, num_users=1):
        """
        Finds the most similar users to the input user and then compares
        the raw data for similar users with the input user, returning any
        items the similar users have rated.

        Input: 
        UserID (int): ID of the user to recommend for
        num_users: The number of closest users to return items from

        Returns: 
        Recs (list): List of recommended item IDs
        """
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        userrecs = []
        for user in range(self.U.shape[0]):
            if user!= userID:
                userrecs.append([user,self.user_similarity(userID,user)])
        final_rec = [i[0] for i in sorted(userrecs,key=lambda x: x[1],reverse=True)]
        comp_user = final_rec[:num_users]
        print("User #%s's most similar user is User #%s "% (userID, comp_user))
        current = self.get_row(userID)
        recs = []
        for user in comp_user:
            rec_likes = self.get_row(user)
            for i,item in enumerate(current):
                if item != rec_likes[i] and rec_likes[i]!=0:
                    recs.append(i)
        return list(set(recs))

