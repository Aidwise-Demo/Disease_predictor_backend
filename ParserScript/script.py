

def process_837_file(file_path):
    import os
    import pandas as pd
    import re
    # Initialize an empty list to store the lines of text
    lines = []
    print("Start")
    # Read the text file line by line with the 'latin-1' encoding
    with open(file_path, "r", encoding='latin-1') as file:  # Specify 'latin-1' encoding here
        for line in file:
            parts = line.split("~")
            cleaned_parts = [part.strip() for part in parts]  # Remove trailing and leading whitespaces
            lines.extend(cleaned_parts)

    # Remove lines that are empty (i.e., became empty after stripping trailing whitespace)
    lines = [line for line in lines if line]
    
    # Create a pandas DataFrame from the list of lines
    df = pd.DataFrame({"Text": lines})

    # Now you can work with the DataFrame, e.g., filter and process the data
    final_list = []
    current_list = []
    
    
    # Iterate through each item in the "Text" column of the DataFrame
    for code in df["Text"]:
        # Check if the code contains "**" and starts with "HL"
        if "**" in code and code.startswith("HL"):
            # If current_list is not empty, add it to final_list
            if current_list:
                final_list.append(current_list)
            # Reset current_list to an empty list
            current_list = []
        # Add the code to current_list
        current_list.append(code)

    # If current_list is not empty, add it to final_list
    if current_list:
        final_list.append(current_list)

    # Initialize an empty list called filter_
    filter_ = []

    # Iterate through each sublist in final_list
    for x in final_list:
        # Filter rows that do not start with certain prefixes
        filter_.append([row for row in x if not (row.startswith('HL') or row.startswith('LX') or row.startswith('SE') or row.startswith('GE') or row.startswith('IEA'))])

    # Sort the filtered data by length in descending order
    Final_data = sorted(filter_, key=len, reverse=True)

    # Create a DataFrame named raw from Final_data
    data = pd.DataFrame(Final_data)

    columns = [
        "provider",
        "provider name",
        "subscriber name",
        "DOB",
        "name_pr",
        "patient name",
        "claim info",
        "diagnosis",
        "doctor name"
    ]
    result_df = pd.DataFrame(columns=columns)

    # Define a function 'extract_data' that takes a row as an argument
    def extract_data(row):
        # Initialize a dictionary to store data elements
        data_dict = {
            "provider": "",
            "provider name": "",
            "subscriber name": "",
            "name_pr": "",
            "patient name": "",
            "claim info": "",
            "doctor name": ""
        }
        # Initialize empty lists for different elements
        sv1_elements = []  # Placeholder for SV1 elements
        dtp_elements = []  # Placeholder for DTP elements
        H_elements = []    # Placeholder for H elements
        Zip_codes = []     # Placeholder for Zip codes
        DMG = []           # Placeholder for DMG (demographic) data

        for column in data.columns:
            cell = row.get(column, "")  
            elements = str(cell).split() 
            for i, element in enumerate(elements):
                if element.startswith("PRV*BI*PXC"):
                    data_dict["provider"] = " ".join(elements[i:])
                elif element.startswith("NM1*85*2"):
                    data_dict["provider name"] = " ".join(elements[i:])
                elif element.startswith("NM1*IL*1"):
                    data_dict["subscriber name"] = " ".join(elements[i:])
                elif element.startswith("DMG*D8"):
                    DMG.append(" ".join(elements[i:]))
                elif element.startswith("NM1*PR"):
                    data_dict["name_pr"] = " ".join(elements[i:])
                elif element.startswith("NM1*QC"):
                    data_dict["patient name"] = " ".join(elements[i:])
                elif element.startswith("CLM"):
                    data_dict["claim info"] = " ".join(elements[i:])
                elif element.startswith("HI*"):
                    H_elements.append(" ".join(elements[i:]))    
                elif element.startswith("NM1*DN"):
                    data_dict["doctor name"] = " ".join(elements[i:])
                elif element.startswith("NM1*77"):
                    data_dict["lab"] = " ".join(elements[i:])
                elif element.startswith("SV1"):
                    sv1_elements.append(" ".join(elements[i:]))
                elif element.startswith("DTP"):
                    dtp_elements.append(" ".join(elements[i:]))
                elif element.startswith("N4"):
                    Zip_codes.append(" ".join(elements[i:]))


        print("122")
        for i, dtp_element in enumerate(dtp_elements):
            data_dict[f"date of diagnosis{i + 1}"] = dtp_element
        # Append multiple "SV1" and "DTP" elements to new columns
        for i,DMG in enumerate(DMG):
            data_dict[f"DMG{i+1}"]=DMG
        for i,H_elements in enumerate(H_elements):
            data_dict[f"diagnosis{i+1}"]=H_elements
        for i, sv1_element in enumerate(sv1_elements):
            data_dict[f"DIscreption{i + 1}"] = sv1_element
        for i, dtp_element in enumerate(dtp_elements):
            data_dict[f"date of diagnosis{i + 1}"] = dtp_element
        for i, Zip_code in enumerate(Zip_codes):
            data_dict[f"Zipcode{i + 1}"] = Zip_code
        return data_dict
    # for index, row in data.iterrows():
    #     data_dict = extract_data(row)
    #     result_df = result_df.append(data_dict, ignore_index=True)

    dfs = []  # Create an empty list to store DataFrames

    for index, row in data.iterrows():
        data_dict = extract_data(row)
        # Convert the dictionary to a DataFrame and append it to the list
        dfs.append(pd.DataFrame([data_dict]))

    # Concatenate the list of DataFrames into a single DataFrame
    result_df = pd.concat(dfs, ignore_index=True)


    result_df = result_df.fillna("")
    new_df = pd.DataFrame()


    # Splitting 'provider' column by '*' and selecting the 4th element for 'BillingProvider'
    new_df["BillingProvider"] = result_df["provider"].str.split('*', expand=True)[3]

    # Splitting 'provider name' column by '*' and selecting the 4th and 10th elements for 'BillingProviderName' and 'BillingProviderID'
    new_df["BillingProviderName"] = result_df['provider name'].str.split('*', expand=True)[3]
    new_df["BillingProviderID"] = result_df['provider name'].str.split('*', expand=True)[9]

    # Splitting 'Zipcode1' column by '*' and selecting the 4th element for 'BillingProviderZipcode'
    new_df["BillingProviderZipcode"] = result_df["Zipcode1"].str.split('*', expand=True)[3]

    # Splitting 'subscriber name' column by '*' and selecting the 4th and 5th elements, concatenating for 'SubscriberName'
    new_df["SubscriberName"] = result_df["subscriber name"].str.split('*', expand=True)[3] + "   " + result_df["subscriber name"].str.split('*', expand=True)[4]

    # Selecting the 10th element for 'MemberID' from 'subscriber name' column split by '*'
    new_df["MemberID"] = result_df["subscriber name"].str.split('*', expand=True)[9]


    new_df["PayerName"]=result_df["name_pr"].str.split('*', expand=True)[3]

    new_df["PayerID"]=result_df["name_pr"].str.split('*', expand=True)[9]

    new_df["Subscriber_Gender"]=result_df["DMG1"].str.split('*', expand=True)[3]

    date_component = result_df["DMG1"].str.split('*', expand=True)[2]
    new_df["Subscriber_DOB"] = pd.to_datetime(date_component, format="%Y%m%d", errors="coerce")
    print("180")
    new_df["PaitentName"] = result_df['patient name'].str.split('*', expand=True)[3]+"   "+result_df['patient name'].str.split('*', expand=True)[4]

    new_df["Paitent_Gender"]=result_df["DMG2"].str.split('*', expand=True)[3]

    date_component = result_df["DMG2"].str.split('*', expand=True)[2]
    new_df["Paitent_DOB"] = pd.to_datetime(date_component, format="%Y%m%d", errors="coerce")

    new_df["ClaimNumber"]=result_df["claim info"].str.split('*', expand=True)[1]
    new_df["ClaimAmount"]=result_df["claim info"].str.split('*', expand=True)[2]
    new_df["POS"]=result_df["claim info"].str.split('*', expand=True)[5].str.split(':',expand=True)[0]


    new_df["DoctorName"]=result_df['doctor name'].str.split('*', expand=True)[4]+" "+ result_df['doctor name'].str.split('*', expand=True)[3]
    new_df["DoctorID"]=result_df['doctor name'].str.split('*', expand=True)[9]


    IDC=pd.DataFrame()
    split_columns = result_df['diagnosis1'].str.split('*', expand=True).iloc[:, 1:]
    split_columns.columns = [f"ICD{i+1}" for i in range(split_columns.shape[1])]
    ICD = split_columns
    new_df = pd.concat([ new_df,ICD], axis=1)

    new_df["ServiceFacility"]=result_df["lab"].str.split('*', expand=True)[3]
    new_df["ServiceFacilityID"]=result_df["lab"].str.split('*', expand=True)[9]

    import pandas as pd
    split_dfs = []
    indices_to_extract = [1, 2]
    i = 1
    for column_name in result_df.columns:
        if column_name.startswith("DIscreption"):
            split_df = result_df[column_name].str.split('*', expand=True)
            selected_columns = split_df.iloc[:, indices_to_extract]
            selected_columns.columns = [f"CPT-{i}", f"ClaimCharge-{i}"]
            split_dfs.append(selected_columns)
            i += 1 
    result_dfs = pd.concat(split_dfs, axis=1)
    new_df = pd.concat([ new_df,result_dfs], axis=1)

    i = 1  # Initialize the counter
    for column_name in result_df.columns:
        if column_name.startswith("date of diagnosis"):
            new_df[f"DateofService{i}"] = result_df[column_name].str.split('*', expand=True)[3]

            # Create corresponding "StartDate_for_Service" and "EndDate_for_Service" columns
            new_df[f"StartDate_for_Service-{i}"] = pd.to_datetime(new_df[f"DateofService{i}"].str.split("-", expand=True)[0], format='%Y%m%d')
            new_df[f"EndDate_for_Service-{i}"] = pd.to_datetime(new_df[f"DateofService{i}"].str.split("-", expand=True)[1], format='%Y%m%d')

            i += 1
    # Drop the "DateofService" columns
    for i in range(1, i):  # Drop columns up to the current counter value
        new_df.drop(columns=f"DateofService{i}", inplace=True)
    new_df["subscriberZipcode"] = result_df["Zipcode2"].str.split('*', expand=True)[3]

    def get_n4_after_nm177(row):
        for i in range(len(row)):
            if row[i] is not None and row[i].startswith("NM1*77"):
                for j in range(i + 1, len(row)):
                    if row[j] is not None and row[j].startswith("N4"):
                        return row[j]
        return None
    new_df['Facility_Zipcode'] = data.apply(get_n4_after_nm177, axis=1)
    new_df['Facility_Zipcode']=new_df["Facility_Zipcode"].str.split('*', expand=True)[3]
    def get_n4_after_nm1qc(row):
        found_nm1 = False
        for cell in row:
            if found_nm1 and str(cell).startswith("N4"): 
                return cell
            if str(cell).startswith("NM1*QC"): 
                found_nm1 = True
        return None 
    new_df['Paitent_Zipcode'] = data.apply(get_n4_after_nm1qc, axis=1)
    new_df['Paitent_Zipcode'] = new_df['Paitent_Zipcode'].str.split('*', expand=True)[3]
    def get_n4_after_nm1pr(row):
        found_nm1 = False
        for cell in row:
            if found_nm1 and str(cell).startswith("N4"): 
                return cell
            if str(cell).startswith("NM1*PR"): 
                found_nm1 = True
        return None 
    new_df['provider_Zip'] = data.apply(get_n4_after_nm1pr, axis=1)
    new_df['provider_Zip'] = new_df['provider_Zip'].str.split('*', expand=True)[3]
    new_df['Paitent_Zipcode'].fillna(new_df['subscriberZipcode'], inplace=True)
    new_df['Paitent_DOB'].fillna(new_df['Subscriber_DOB'], inplace=True)
    new_df['PaitentName'].fillna(new_df['SubscriberName'], inplace=True)
    new_df['Paitent_Gender'].fillna(new_df['Subscriber_Gender'], inplace=True)
    def extract_bht(row):
        for cell in row:
            if str(cell).startswith("BHT"):
                return str(cell).split('*')[4] if len(str(cell).split('*')) > 4 else None
    new_df['submission date'] = data.apply(extract_bht, axis=1)
    non_null_value = new_df['submission date'].dropna().iloc[0]  # Get the non-null value
    new_df['submission date'].fillna(non_null_value, inplace=True)
    new_df['submission date'] = pd.to_datetime(new_df['submission date'], format='%Y%m%d')
    specified_prefixes = ['CPT', 'ClaimCharge', 'StartDate_for_Service', 'EndDate_for_Service']
    filtered_columns = [col for col in new_df.columns if not col.startswith(tuple(specified_prefixes))]
    cpt_columns = [col for col in new_df.columns if col.startswith('CPT')]
    n=len(cpt_columns)
    my_dict = {}
    for i in range(1, n+1):
        # Select columns that are not part of specified prefixes and concatenate with specific strings for each iteration 'i'
        selected_columns = filtered_columns + [f'CPT-{i}', f'ClaimCharge-{i}', f'StartDate_for_Service-{i}', f'EndDate_for_Service-{i}']
        my_dict[f'df_{i}'] = new_df[selected_columns]
    modified_dfs = []
    for i in range(1, n+1):
        df_name = f'df_{i}'
        new_column_names = {
            f'CPT-{i}': 'CPT',
            f'ClaimCharge-{i}': 'ClaimCharge',
            f'StartDate_for_Service-{i}': 'StartDate_for_Service',
            f'EndDate_for_Service-{i}': 'endDate_for_Service'
        }
        print("295")
        my_dict[df_name].rename(columns=new_column_names, inplace=True)
        modified_dfs.append(my_dict[df_name])
    concatenated_df = pd.DataFrame()
    concatenated_df = pd.concat(modified_dfs, ignore_index=True)
    concatenated_df = concatenated_df[concatenated_df['CPT'].notnull()]
    concatenated_df['CPT_code'] = concatenated_df['CPT'].apply(lambda x: x.split(":")[1] if isinstance(x, str) else None)
    concatenated_df['CPT_description'] = concatenated_df['CPT'].apply(lambda x: x.split(":")[-2] if isinstance(x, str) else None)
    concatenated_df = concatenated_df.drop(columns=["CPT"])
    # k=file_path.split('\\')[-1].split('.')[0]
    # excel_file = f'output/{k}.xlsx' # Name your Excel file
    from datetime import datetime
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    excel_file = f"output/{formatted_datetime}.xlsx"
    concatenated_df.to_excel(excel_file, index=False)
    print("END")# Save DataFrame as Excel file
    return excel_file




