# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 23:30:26 2021

@author: emile
"""
#import de libraires standards

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# working directory à changer ici, les 3 fichiers et le script doivent etre
# ensemble

dirname = os.getcwd()

# path strings pour les 3 fichiers excel à disposition pour l'exercice

GLEIF_path, TR1_path, TR2_path = os.path.join(dirname, r'GLEIF.xlsx'),\
os.path.join(dirname, r'TR1.xlsx'), os.path.join(dirname, r'TR2.xlsx')

# import .xls comme dataframes

GLEIF_df,TR1_df,TR2_df = pd.read_excel(GLEIF_path), pd.read_excel(TR1_path),\
pd.read_excel(TR2_path)

#concatener TR1 and TR2, et je retire les UTI en double pour garder une seule 
#déclaration si les deux parties ont déclaré aux TR

TR_df = pd.concat([TR1_df,TR2_df]).drop_duplicates(subset=['uti']).\
    reset_index(drop=True)

# ajout des données GLEIF: je conserve les LEI  de toutes les transactions et
# contreparties, mais je supprime les doublons ayant le même UTi et meme
# provenance car ces doublons fausseront les calculs des positions par la suite
    
master_df = pd.concat([GLEIF_df.merge(TR_df, left_on='lei', right_on='lei_othr'), 
                     GLEIF_df.merge(TR_df, left_on='lei', right_on='lei_rptg')])\
    .drop_duplicates(subset=['uti','country']).reset_index(drop=True)


    
# calcul des indicateurs aggregés où je rajoute les positions avec un signe 
# selon le sens de transaction ('sign notional')

master_df['sign notional'] = np.where(master_df['side']=='B',
                                      -master_df['notional'],
                                      master_df['notional'])
result=pd.DataFrame()
result['gross exposure'] = master_df.groupby(["flt", "country"])['notional'].sum()
result['net exposure'] = master_df.groupby(["flt", "country"])['sign notional'].sum()
result['ratio'] = result['net exposure']/result['gross exposure']

# affichage top et bottom 5 pour chaque taux 

print("EONIA bottom 5: \n",result.loc['EONIA']['ratio'].nsmallest(5),
      "\n")
print("EONIA top 5: \n", result.loc['EONIA']['ratio'].nlargest(5),
      "\n")
print("LIBOR bottom 5: \n", result.loc['LIBOR']['ratio'].nsmallest(5),
      "\n")
print("LIBOR top 5: \n", result.loc['LIBOR']['ratio'].nlargest(5))


plt.figure(1)

pd.concat([result.loc['LIBOR']['ratio'].nsmallest(5),
          result.loc['LIBOR']['ratio'].nlargest(5)]).sort_values().plot.bar(ylabel='ratios',
          title='LIBOR exposure ratios top 5 (RHS) and bottom 5 (LHS)',
          color='teal')
                                                                    
plt.figure(2) 
                                                                          
pd.concat([result.loc['EONIA']['ratio'].nsmallest(5),
          result.loc['EONIA']['ratio'].nlargest(5)]).sort_values().plot.bar(ylabel='ratios',
          title='EONIA exposure ratios top 5 (RHS) and bottom 5 (LHS) ',
          color='indianred')













    


