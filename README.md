# Sailing the chemical space towards effective lead identification.

Vast chemical data is housed in chemical public databases, like PubChem, in two-dimensional, string forms which have paved the way for the use of neural networks in the herculean task of safe and effective drug design. 
The NorthStar pipeline connects 2 dimensional chemical data to a deep learning variational autoencoder and synthesizes the pool of lead candidates that will, in the end, be assessed in silico using docking methods and generate consensus pharmacophores.  

Chemical data remains 2 dimensional, a string, a word, with chemical significance, and in a prime form to be used in a neural network.
The model learns, effectively, to map the chemical strings and their properties - lipophilicity and others - to the latent space, creating clusters of structurally dissimilar drugs that, neverthelesss, exhibit the same properties.
The generation of lead candidates is not a result of random sampling of the latent space, rather a guided generation and exploration of that latent chemical space, using the "guide", the North Star, a.k.a a "golden" drug, known to effectively work against our Target of interest.

My North Star:

Remdesivir, a well-known [antiviral drug](https://www.google.com/search?q=remdesivir&oq=remdesivir&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiABDIHCAIQABiABDIHCAMQABiABDIHCAQQABiABDIHCAUQABiABDIHCAYQABiABDIHCAcQABiABDIHCAgQABiABDIHCAkQABiPAtIBCDE4NjhqMGo5qAIAsAIB&sourceid=chrome&ie=UTF-8) targeting the RNA-dependent RNA polymerase enzyme, a key Target when designing antiviral drugs. 

![800px-Remdesivir svg](https://github.com/user-attachments/assets/6f5e02ed-188f-4be2-9731-8d09f14df1f1)

The result: constellations, drugs that will exist in this latent space, close to the North Star's "area" of properties and effects, but dissimilar in terms of structure. 

Two candidate leads, in the three-dimensional pocket of the enzyme that makes up the core of its activity:

![image](https://github.com/user-attachments/assets/9892dadf-899a-4f0a-8be1-4ffae0685ee6)

The pipeline:

![image](https://github.com/user-attachments/assets/1bf9b223-c2e1-4394-a4fa-659038ff0079)

The architecture:
![image](https://github.com/user-attachments/assets/0411341a-5e33-4b3e-8c0c-e0b1ac4e6e4f)
