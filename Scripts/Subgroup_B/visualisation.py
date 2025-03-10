# Referring to google forms, we can see that Revenge of the Mummy (ROTM) > Cylon > Transformers were the most popular rides, "Thrilling"
# However, these 3 rides also had the longest wait times among all the rides 
# Additionally, Transformers, Sesame Street, and ROTM had the most complaints (18,18,13 out of 210 respectively), "Boring", "Too scary (drop)"
# But the average consensus on the wait times being worth it was a meh 3/5'

import pandas as pd
uss_df = pd.read_csv('USS_Responses.csv')
print(uss_df.head()) # 55 columns