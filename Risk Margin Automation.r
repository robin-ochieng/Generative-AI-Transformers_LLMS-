# Importing the libraries
library(tidyverse)
library(tidyverse)
library(readxl)
library(rio)
library(dplyr)
library(tidyr)
library(ChainLadder)
library(lubridate)
library(zoo)
library(readr)
library(writexl)

#Loading in the Dataset
Insurer_1 <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                        sheet = "Insurer 1")

Insurer_1_ <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                         sheet = "Insurer 1_")

Insurer_1_2 <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                          sheet = "Insurer 1_2")

Insurer_2 <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                        sheet = "Insurer 2 ")

Insurer_2_ <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                         sheet = "Insurer 2_")

Insurer_2_3 <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                          sheet = "Insurer 2_3")

Insurer_3 <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                        sheet = "Insurer 3")

Insurer_3_ <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                         sheet = "Insurer 3_")

Insurer_3__ <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                          sheet = "Insurer 3__")

Insurer_3_4_5_6 <- read_excel("C:/Users/Robin Ochieng/Kenbright/Automation and Risk Margin - General/Tanzania/TIRA Risk Margin/Data/Data Final/Medical_Claims.xlsx", 
                              sheet = "Insurer 3_4_5_6")

#Merging the Data by rows 
data = rbind(Insurer_1, Insurer_1_, Insurer_1_2, Insurer_2, Insurer_2_, Insurer_2_3, Insurer_3, Insurer_3_,  Insurer_3__,  Insurer_3_4_5_6) 

#Filter for Paid Claims
data <- data %>%
  filter(`CLAIM STATUS` == 'Outstanding')

# Now perform the rest of the operations on the data
Val_Data_MD <- data %>%
  #filter(`Main Class of Business` %in% Val_class)%>%
  summarise(Gross_Amount = sum(`CLAIM AMOUNT`, na.rm = T))