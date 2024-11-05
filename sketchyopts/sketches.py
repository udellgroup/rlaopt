import jax.numpy as jnp
import jax.random as jrn
from sketchyopts.base import KeyArray
from typing import Tuple

""" Module containing functions for computing various types of sketches """

####### Gaussian ####### 
def gauss_sketch(key: KeyArray,
                 A: jnp.ndarray, 
                 s: int, 
                 mode: str = 'left')-> Tuple[jnp.ndarray, KeyArray]:
    key, subkey = jrn.split(key)
    if mode == 'left':
       n = A.shape[0]
       S = jrn.normal(subkey, (s,n))/jnp.sqrt(s)
       SA = S@A
       return SA, key
    
    else:
       p = A.shape[1]
       S = jrn.normal(subkey, (p,s))
       AS = A@S
       return AS, key

def gauss_sketch_mat(key: KeyArray,
                     op_shape: tuple[int, int], 
                     s: int, 
                     mode: str = 'left')-> Tuple[jnp.ndarray, KeyArray]:
      
      key, subkey = jrn.split(key)
      
      if mode == 'left':
         n = op_shape[0]
         S = jrn.normal(subkey, (s,n))/jnp.sqrt(s)
         return S, key
    
      else:
         p = op_shape[1]
         S = jrn.normal(subkey, (p,s))
      return S, key



####### Ortho ####### 
def ortho_sketch(key: KeyArray,
                 A: jnp.ndarray,
                 s: int,
                 mode: str)-> Tuple[jnp.ndarray, KeyArray]:
   
   key, subkey = jrn.split(key)
   
   if mode == 'left':
      n = A.shape[0]
      G = jrn.normal(subkey, (s, n))/jnp.sqrt(s)
      S, _ = jnp.linalg.qr(G, mode="reduced")
      SA = S@A
      return SA, key
   
   else:
      p = A.shape[1]
      G = jrn.normal(subkey, (p, s))/jnp.sqrt(p)
      S, _ = jnp.linalg.qr(G, mode="reduced")
      AS = A@S
      return AS, key

def ortho_sketch_mat(key: KeyArray,
                     op_shape: tuple[int, int],
                     s: int,
                     mode: str)-> Tuple[jnp.ndarray, KeyArray]:
   
   key, subkey = jrn.split(key)
   
   if mode == 'left':
      n = op_shape[0]
      G = jrn.normal(subkey, (s, n))/jnp.sqrt(s)
      S, _ = jnp.linalg.qr(G, mode="reduced")
      return S, key
   
   else:
      p = op_shape[1]
      G = jrn.normal(subkey, (p, s))/jnp.sqrt(p)
      S, _ = jnp.linalg.qr(G, mode="reduced")
      return S, key


####### SJLT ####### 
def sjlt_sketch(key: KeyArray,
                A: jnp.ndarray, 
                s: int,
                mode: str ='left')->tuple[jnp.ndarray, KeyArray]:
   
    key, subkey_1, subkey_2 = jrn.split(key, 3)
   
    if mode == 'left':
        n = A.shape[0]  # Number of columns
        
        if s>=8:
           zeta=8
        else:
           zeta=s
        
        # Initialize S as a zero matrix
        S = jnp.zeros((s, n))
        
        # Generate random +1/-1 values for zeta entries in each column
        b = jrn.bernoulli(subkey_1, shape=(zeta, n))  # Bernoulli random variables
        z = 2 * b - 1  # Convert to +1/-1
        
        # Generate random row indices for each non-zero entry in each column
        row_indices = jrn.choice(subkey_2, s, shape=(zeta, n), replace=True)
        
        # Vectorized scatter update to place z values in S at the appropriate row indices
        S = (S.at[row_indices, jnp.arange(n)].set(z))*jnp.sqrt(1/zeta)
        SA = S@A
        return SA, key
    else:
       p = A.shape[1]
       if s>=8:
           zeta=8
       else:
           zeta=s
        
        
       # Initialize S as a zero matrix
       S = jnp.zeros((p, s))
        
       # Generate random +1/-1 values for zeta entries in each column
       b = jrn.bernoulli(subkey_1, shape=(zeta, p))  # Bernoulli random variables
       z = 2 * b - 1  # Convert to +1/-1
        
       # Generate random col indices for each non-zero entry in each row
       col_indices = jrn.choice(subkey_2, s, shape=(zeta, p), replace=True)
        
       # Vectorized scatter update to place z values in S at the appropriate column indices
       S = (S.at[jnp.arange(p), col_indices].set(z))
       AS = A@S
       return AS, key

def sjlt_sketch_mat(key: KeyArray,
                    op_shape: tuple[int, int], 
                    s: int,
                    mode: str ='left'
                   )->tuple[jnp.ndarray, KeyArray]:
   
    key, subkey_1, subkey_2 = jrn.split(key, 3)
   
    if mode == 'left':
        n = op_shape[0]  # Number of rows of A
        
        if s>=8:
           zeta=8
        else:
           zeta=s
        
        # Initialize S as a zero matrix
        S = jnp.zeros((s, n))
        
        # Generate random +1/-1 values for zeta entries in each column
        b = jrn.bernoulli(subkey_1, shape=(zeta, n))  # Bernoulli random variables
        z = 2 * b - 1  # Convert to +1/-1
        
        # Generate random row indices for each non-zero entry in each column
        row_indices = jrn.choice(subkey_2, s, shape=(zeta, n), replace=True)
        
        # Vectorized scatter update to place z values in S at the appropriate row indices
        S = (S.at[row_indices, jnp.arange(n)].set(z))*jnp.sqrt(1/zeta)

        return S, key
    else:
       p = op_shape[1] # Number of cols of A
       if s>=8:
           zeta=8
       else:
           zeta=s
        
        
       # Initialize S as a zero matrix
       S = jnp.zeros((p, s))
        
       # Generate random +1/-1 values for zeta entries in each column
       b = jrn.bernoulli(subkey_1, shape=(zeta, p))  # Bernoulli random variables
       z = 2 * b - 1  # Convert to +1/-1
        
       # Generate random col indices for each non-zero entry in each row
       col_indices = jrn.choice(subkey_2, s, shape=(zeta, p), replace=True)
        
       # Vectorized scatter update to place z values in S at the appropriate column indices
       S = (S.at[jnp.arange(p), col_indices].set(z))
       return S, key
         
