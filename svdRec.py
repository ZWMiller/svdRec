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
        self.U, self.s, self.V = (None,None,None)
        self.user_encoder = None
        self.item_encoder = None
        self.mat = None
        self.decomp = False
    
    def load_csv_sparse(self,filename,delimiter=',',skiprows=None):
        print("Note: load_csv_sparse expects a csv in the format of: rowID, colID, Value, ...")
        u, m, r = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows, usecols=(0,1,2)).T
        self.mat = coo_matrix((r, (u-1, m-1)), shape=(u.max(), m.max())).tocsr()
        print("Created matrix of shape: ",self.mat.shape)
        
    def load_data_numpy(self, array, data_type=float):
        self.mat = csr_matrix(array,dtype=data_type)
        print("Created matrix of shape: ",self.mat.shape)
        
    def load_item_encoder(self, d):
        if type(d) != dict:
            raise TypeError("Encoder must be dictionary with key = itemID and value = Title")
        self.item_encoder = d
        
    def load_user_encoder(self, d):
        if type(d) != dict:
            raise TypeError("Encoder must be dictionary with key = userID and value = Title")
        self.user_encoder = d
        
    def get_item_name(self,itemid):
        if self.item_encoder:
            return self.item_encoder[str(itemid)]
        else:
            return "No ItemId -> Item-name Encoder Built!"
    
    def get_user_name(self,userid):
        if self.item_encoder:
            return self.user_encoder[str(userid)]
        else:
            return "No UserID -> Username Encoder Built!"
    
    def SVD(self, num_dim=None):
        if num_dim==None:
            print("Number of SVD dimensions not requested, using %s dimensions." % (min(self.mat.shape)-1), "To set, use num_dim.")
            num_dim = min(self.mat.shape)-1
        self.U, self.s, self.VT = svds(self.mat,k=num_dim)
        self.decomp = True
    
    def get_cell(self,i,j):
        return self.mat[1,:].toarray()[0,j]
    
    def get_similar_items(self, itemID, num_recom=5, show_similarity=False):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        recs = []
        for item in range(self.VT.T.shape[0]):
                recs.append([item+1,self.item_similarity(itemID-1,item)])
        if show_similarity:
            final_rec = [(i[0],i[1]) for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        else:
            final_rec = [i[0] for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        return final_rec[:num_recom]
    
    def item_similarity(self,item1,item2):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.VT.T[item1],self.VT.T[item2])
    
    def user_similarity(self,user1,user2):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.U[user1],self.U[user2])
    
    def user_item_similarity(self,user,item):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.U[user],self.VT.T[item])
    
    def user_item_predict(self,user,item):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.U[user],self.VT.T[item])
        
    def recommends_for_user(self, userID, num_recom=2, show_similarity=False):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        recs = []
        for item in range(self.VT.T.shape[0]):
            recs.append((item+1,self.user_item_predict(userID-1,item)))
        if show_similarity:
            final_rec = [(i[0],i[1]) for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        else:
            final_rec = [i[0] for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        return final_rec[:num_recom]
    
    def recs_from_closest_user(self, userID, num_users=1):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        userrecs = []
        for user in range(self.U.shape[0]):
            if user!= userID:
                userrecs.append([user,self.user_similarity(userID,user)])
        final_rec = [i[0] for i in sorted(userrecs,key=lambda x: x[1],reverse=True)]
        comp_user = final_rec[:num_users]
        print("User #%s's most similar user is User #%s "% (userID, comp_user))
        data = self.mat.toarray()
        current = data[userID]
        recs = []
        for user in comp_user:
            rec_likes = data[user]
            for i,item in enumerate(current):
                if item != rec_likes[i] and rec_likes[i]!=0:
                    recs.append(i)
        return list(set(recs))

