def identify_data_type(data):
    if data.isdigit():
        return "Numeric"
    elif data.isalpha():
        return "Alphabetic"
    elif data.isalnum():
        return "Alphanumeric"
    else:
        return "Other"
print("Nmae: Masruk Habib, Class:7TC-1, Enrollment:92100103165, Lab: C")
def main():
    # Get input from the user
    user_input = input("Enter different types of data separated by spaces: ")
    
    # Split the input into individual items
    data_items = user_input.split()
    
    # Identify and display the type of each item
    for item in data_items:
        data_type = identify_data_type(item)
        print(f"'{item}' is of type: {data_type}")
if __name__ == "__main__":
    main()
# need to delete 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Calculate correlation matrix
corr_matrix = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(15, 8))
plt.title('Correlation Matrix', fontsize=11)

# Create a heatmap
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": 14})

# Add aesthetics
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
# Show plot
plt.show()
