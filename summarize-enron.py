# Add your python script here
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 144

import pandas as pd
import numpy as np
import zipfile
from datetime import datetime
from pandas.plotting import table
import itertools
import os
import shutil

import matplotlib.pyplot as plt

with zipfile.ZipFile('RedOwl-Data-Science-Recruiting-Exam-public-master.zip', 'r') as z:
    # Extract enron-event-history-all.csv
    for item in z.namelist():
        if item == 'RedOwl-Data-Science-Recruiting-Exam-public-master/enron-event-history-all.csv':
            csv_data = z.open('RedOwl-Data-Science-Recruiting-Exam-public-master/enron-event-history-all.csv')

            rows = csv_data.readlines()

            rows = list(map(lambda x: x.strip(), rows))

            data = []

            for row in rows:
                # further split each row into columns assuming delimiter is comma
                row = row.decode().split(',')

                # append to data-frame our new row-object with columns
                data.append(row)

# Restructure list 'data' and convert data.time column to proper dtype
data = pd.DataFrame(data)
data.drop(data.iloc[:, 6:10], inplace=True, axis=1)
data.columns = ["time", "message identifier", 'sender', 'recipients', 'topic', 'mode']
data.time = [np.datetime64(i.replace('"','')).view('<i8') for i in data.time]
data.time = [datetime.fromtimestamp(i/1000.0) for i in data.time]
data.time = [str(i.replace(microsecond=0)) for i in data.time]

# Function that returns a dataframe of people sorted by most sent e-mails
def person_data(data):
    # Groupby 'sender' and 'recipients' and merge dataframes together
    sent_data = data.groupby("sender").size().to_frame()
    sent_data.columns = ["sent"]
    received_data = data.groupby("recipients").size().to_frame()
    received_data.columns = ["received"]
    new_df = pd.merge(sent_data, received_data, left_index=True, right_index=True)

    # Sort dataframe by 'sent
    new_df = new_df.sort_values('sent', ascending=False)

    # Write dataframe to csv
    return new_df.to_csv('E-mail_Summary_By_Person.csv', index=True, header=True, sep=',', encoding='utf-8')

# Function that flips legend from vertical to horizontal
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def plot_data(data, top):
    # Groupby 'sender' and 'recipients' and merge dataframes together
    sent_data = data.groupby("sender").size().to_frame()
    sent_data.columns = ["sent"]
    received_data = data.groupby("recipients").size().to_frame()
    received_data.columns = ["received"]
    new_df = pd.merge(sent_data, received_data, left_index=True, right_index=True)
    new_df = new_df.sort_values('sent', ascending=False)

    # Obtain list of names of top number of people
    name_list = new_df.head(top).index.tolist()
    # Subset dataframe by list of names
    subset_data = data[data['sender'].isin(name_list)].sort_values('time', ascending=True)[["time", "sender"]]

    # Plot by month
    plt.rcParams["figure.figsize"] = [10, 10]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3) # space out heights of subplots
    ax1 = fig.add_subplot(2, 2, 1) # Define location of subplot
    ax1.set_title("Number of E-mails Sent Over Time (By Months)", fontsize=10) # Define title
    ax1.set_ylabel("Frequency") # Define y-axis label

    by_month_df = subset_data
    # Convert string to months only
    by_month_df['Month'] = pd.to_datetime(by_month_df['time']).dt.to_period('m')
    # Groupby 'sender' and 'month' to obtain each month's e-mail frequency by sender
    by_month_plot = by_month_df.groupby(['sender', 'Month']).size().to_frame().reset_index()
    # Rename column
    by_month_plot.rename(columns={by_month_plot.columns[2]: "Frequency"}, inplace=True)
    # Create cumulative frequency column (Not used)
    #by_month_plot['Cum_Sum'] = by_month_plot['Frequency'].cumsum()

    # Plot multiple line plot representing each sender
    month_plots = [(group.plot(x='Month', y='Frequency', ax=ax1, label=n, legend=False)) for n, group in
                   by_month_plot.groupby('sender')]

    # Plot by day of the year
    ax2 = fig.add_subplot(2, 2, 2) # Define location of subplot
    ax2.set_title("Number of E-mails Sent Over Time (By Day of the Year)", fontsize=10)# Define title
    ax2.set_ylabel("Frequency")# Define y-axis label

    by_dayofyear_df = subset_data
    # Convert string to day of year (out of 365)
    by_dayofyear_df['Day of Year'] = pd.to_datetime(by_dayofyear_df['time'])
    by_dayofyear_df['Day of Year'] = [int(format(dt, '%j')) for dt in by_dayofyear_df['Day of Year']]
    # Groupby 'sender' and 'day of year' to obtain each month's e-mail frequency by sender
    dayofyear_plot = by_month_df.groupby(['sender', 'Day of Year']).size().to_frame().reset_index()
    # Rename column
    dayofyear_plot.rename(columns={dayofyear_plot.columns[2]: "Frequency"}, inplace=True)
    # Create cumulative frequency column (Not used)
    #dayofyear_plot['Cum_Sum'] = dayofyear_plot['Frequency'].cumsum()

    # Plot multiple line plot representing each sender
    dayofyear_plots = [(group.plot(x='Day of Year', y='Frequency', ax=ax2, label=n, legend=False)) for n, group in
                       dayofyear_plot.groupby('sender')]

    # Plot by hour and minute
    ax3 = fig.add_subplot(2, 2, 3) # Define location of subplot
    ax3.set_title("Number of E-mails Sent Over Time (By Hour/Minute)", fontsize=10)
    ax3.set_ylabel("Frequency")

    hourminute_df = subset_data
    # Convert string to Hour/Minute format
    hourminute_df['Hour/Minute'] = pd.to_datetime(hourminute_df['time'])
    hourminute_df['Hour/Minute'] = [i.strftime('%H:%M') for i in hourminute_df['Hour/Minute']]
    # Groupby 'sender' and 'Hour/Minute' to obtain each month's e-mail frequency by sender
    hourminute_plot = hourminute_df.groupby(['sender', 'Hour/Minute']).size().to_frame().reset_index()
    # Rename column
    hourminute_plot.rename(columns={hourminute_plot.columns[2]: "Frequency"}, inplace=True)
    # Create cumulative frequency column (Not used)
    # hourminute_plot['Cum_Sum'] = hourminute_plot['Frequency'].cumsum()

    # Plot multiple line plot representing each sender
    hourminute_plots = [(group.plot(x='Hour/Minute', y='Frequency', ax=ax3, label=n, legend=False)) for n, group in
                        hourminute_plot.groupby('sender')]

    # Plot Table
    ax4 = fig.add_subplot(2, 2, 4) # Define location of subplot
    ax4.xaxis.set_visible(False)  # hide the x axis
    ax4.yaxis.set_visible(False)  # hide the y axis
    ax4.spines['top'].set_visible(False) # hide the top spine of plot
    ax4.spines['right'].set_visible(False) # hide the right spine of plot
    ax4.spines['left'].set_visible(False) # hide the left spine of plot
    ax4.spines['bottom'].set_visible(False) # hide the bottom spine of plot

    # Create a table plot
    table_plot = table(ax4, new_df.head(top), loc='center right', colWidths=[0.3 for x in new_df.columns])

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(flip(handles, top), flip(labels, top), ncol=top, loc='lower center')

    # Return subplots and save plot as png
    #return month_plots, dayofyear_plots, hourminute_plots, table_plot, fig.savefig('Q2.png')

    fig.savefig('Q2.png')


def relative_frequency_plot(data, top):
    # Groupby 'sender' and 'recipients' and merge dataframes together
    sent_data = data.groupby("sender").size().to_frame()
    sent_data.columns = ["sent"]
    received_data = data.groupby("recipients").size().to_frame()
    received_data.columns = ["received"]
    new_df = pd.merge(sent_data, received_data, left_index=True, right_index=True)
    new_df = new_df.sort_values('sent', ascending=False)

    # Obtain list of names of top number of people
    name_list = new_df.head(top).index.tolist()
    # Subset dataframe by list of names
    subset_data = data[data['recipients'].isin(name_list)].sort_values('time', ascending=True)[
        ["time", "sender", "recipients"]]

    # Plot by month
    plt.rcParams["figure.figsize"] = [10, 10]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3) # space out height of subplots
    ax1 = fig.add_subplot(2, 2, 1) # define location of subplot
    ax1.set_title("Relative Frequency of E-mails Sent Over Time (By Months)", fontsize=10) # Define Title
    ax1.set_ylabel("Relative Frequency") # Define y-axis label

    month_df = subset_data
    # Convert string to months only
    month_df['Month'] = pd.to_datetime(month_df['time']).dt.to_period('m')
    # Obtain number of unique senders by month
    month_df['total_unique'] = month_df['sender'].nunique()
    # Groupby 'recipients' and 'month' to obtain each month's e-mail relative frequency by sender
    month_plot = (month_df.groupby(['recipients', 'Month']).apply(
        lambda x: x['sender'].nunique()) / len(month_df['total_unique'])).to_frame().reset_index()
    # Rename Column
    month_plot.rename(columns={month_plot.columns[2]: "Relative Frequency"}, inplace=True)

    # Plot multiple line plot representing each recipient
    month_plots = [(group.plot(x='Month', y='Relative Frequency', ax=ax1, label=n, legend=False)) for n, group in
                   month_plot.groupby('recipients')]

    # Plot by dayofyear
    ax2 = fig.add_subplot(2, 2, 2) # Define location of each subplot
    ax2.set_title("Relative Frequency of E-mails Sent Over Time (By Day of Year)", fontsize=10) # Define Title
    ax2.set_ylabel("Relative Frequency") # Define y-axis label

    dayofyear_df = subset_data
    # Convert string to day of year format (out of 365 days)
    dayofyear_df['Day of Year'] = pd.to_datetime(dayofyear_df['time'])
    dayofyear_df['Day of Year'] = [int(format(dt, '%j')) for dt in dayofyear_df['Day of Year']]
    # Obtain number of unique senders by day of the year
    dayofyear_df['total_unique'] = dayofyear_df['sender'].nunique()
    # Groupby 'recipients' and 'Day of Year' to obtain each month's e-mail relative frequency by sender
    dayofyear_plot = (dayofyear_df.groupby(['recipients', 'Day of Year']).apply(
        lambda x: x['sender'].nunique()) / len(dayofyear_df['total_unique'])).to_frame().reset_index()
    # Rename Column
    dayofyear_plot.rename(columns={dayofyear_plot.columns[2]: "Relative Frequency"}, inplace=True)

    # Plot multiple line plot representing each recipient
    dayofyear_plots = [(group.plot(x='Day of Year', y='Relative Frequency', ax=ax2, label=n, legend=False)) for n, group in
                       dayofyear_plot.groupby('recipients')]

    # Plot by hour and minute
    ax3 = fig.add_subplot(2, 2, 3) # Define subplot location
    ax3.set_title("Relative Frequency of E-mails Sent Over Time (By Hour/Minute)", fontsize=10) # Define Title
    ax3.set_ylabel("Relative Frequency") # Define y-axis label

    hourminute_df = subset_data
    # Convert string to Hour/Minute Format
    hourminute_df['Hour/Minute'] = pd.to_datetime(hourminute_df['time'])
    hourminute_df['Hour/Minute'] = [i.strftime('%H:%M') for i in hourminute_df['Hour/Minute']]
    # Obtain number of unique senders by hour and minute
    hourminute_df['total_unique'] = hourminute_df['sender'].nunique()
    # Groupby 'recipients' and 'Hour/Minute to obtain each month's e-mail relative frequency by sender
    hourminute_plot = (hourminute_df.groupby(['recipients', 'Hour/Minute']).apply(
        lambda x: x['sender'].nunique()) / len(hourminute_df['total_unique'])).to_frame().reset_index()
    # Rename Column
    hourminute_plot.rename(columns={hourminute_plot.columns[2]: "Relative Frequency"}, inplace=True)
    # Plot multiple line plot representing each recipient
    hourminute_plots = [(group.plot(x='Hour/Minute', y='Relative Frequency', ax=ax3, label=n, legend=False)) for n, group in
                        hourminute_plot.groupby('recipients')]

    # Plot Table
    ax4 = fig.add_subplot(2, 2, 4) # Define location of table
    ax4.xaxis.set_visible(False)  # hide the x axis
    ax4.yaxis.set_visible(False)  # hide the y axis
    ax4.spines['top'].set_visible(False) # hide the top spline
    ax4.spines['right'].set_visible(False) # hide the right spline
    ax4.spines['left'].set_visible(False) # hide the left spline
    ax4.spines['bottom'].set_visible(False) # hide the bottom spline

    # Create a table plot
    table_plot = table(ax4, new_df.head(top), loc='center right', colWidths=[0.3 for x in new_df.columns])

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(flip(handles, top), flip(labels, top), ncol=top, loc='lower center')

    # Return subplots and save plot as png
    # return month_plots, dayofyear_plots, hourminute_plots, table_plot, fig.savefig('Q3.png')

    fig.savefig('Q3.png')

# Place outputs in folder and zip folder
def create_and_zip_folder():
    # Define current path
    source = os.getcwd() + '\\'
    # Define Destination Folder
    destination = source + 'Output_Folder'

    # Check if folder exists
    # If folder doesn't exists, create the folder
    if os.path.exists(destination) is False:
        os.mkdir(destination)
    # If folder exists, check if the folder is empty
    elif os.path.exists(destination) is True:
        # If folder isn't empty, delete folder
        if os.stat(destination).st_size == 0:
            shutil.rmtree(destination)

    # Create list of file names
    files = ['E-mail_Summary_By_Person.csv', 'Q2.png', 'Q3.png']

    # Loop through list of file names and store files into folder
    for file in files:
        shutil.move(source + file, destination)

    # Zip folder
    shutil.make_archive('Output', 'zip', root_dir=destination)
    shutil.rmtree(destination)

person_data(data)
plot_data(data, top=3)
relative_frequency_plot(data, top=3)
create_and_zip_folder()

with zipfile.ZipFile('Output.zip', 'r') as z:
    # Print items in zipped folder
    for item in z.namelist():
        print(item)
