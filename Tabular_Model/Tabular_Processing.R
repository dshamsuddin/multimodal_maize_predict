##############################################################################################################################
####################################################2017 & 2018 PROCESSING####################################################
##############################################################################################################################

#NOTE: 2017 & 2018 SAMPLES WERE PROCESSED BELOW. 2019 WAS HANDLED DIFFERENTLY AND CODE CAN BE SEEN AFTER PROCESSING OF 2017/2018.

# Load necessary libraries
library(dplyr)
library(tidyr)
library(lubridate)
library(reshape2)


# Read in the data
data <- read.csv("2017_2018_data.csv")

# Extract the necessary columns
data_filtered <- data[, c("Env", "Date", "T2MDEW", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN", "PRECTOTCORR")]

# Convert the date column
data_filtered$Date <- as.Date(as.character(data_filtered$Date), format = "%Y%m%d")

# Define the initial dates and dataframe names (Modify to include any other environments)
plant_dates <- c("DG2F_2017" = "2017-03-03", "G2FE_2017" = "2017-03-03", 
                 "G2LA_2017" = "2017-04-06", "G2FN_2018" = "2018-03-07", 
                 "G2FE_2018" = "2018-03-08", "G2LA_2018" = "2018-04-03")

# Create empty data frames for each environment
DG2F_2017 <- data.frame()
G2FE_2017 <- data.frame()
G2LA_2017 <- data.frame()
G2FN_2018 <- data.frame()
G2FE_2018 <- data.frame()
G2LA_2018 <- data.frame()

# Iterate through each environment and calculate the aggregations for every week
for (env in names(plant_dates)) {
  start_date <- as.Date(plant_dates[env])
  
  # Calculate week number from -13 to any weeks you want relative to the start date
  data_env <- data_filtered %>%
    filter(Date >= (start_date - weeks(13)), Date < (start_date + weeks(14))) %>% 
    mutate(Week = ((row_number() - 1) %/% 7) - 13) %>%
    group_by(Week) %>%
    summarise(T2MDEW = sum(T2MDEW),
              RH2M = mean(RH2M),
              WS2M = sum(WS2M),
              ALLSKY_SFC_SW_DWN = sum(ALLSKY_SFC_SW_DWN),
              T2M_MAX = mean(T2M_MAX),
              T2M_DIFF_SUM = sum(T2M_MAX - T2M_MIN),
              T2M_GDD = sum(((T2M_MAX + T2M_MIN)/2)-10),
              T2M_MIN = mean(T2M_MIN),
              PRECTOTCORR = sum(PRECTOTCORR)) %>%
    ungroup()

         
  # Assign the result to the corresponding data frame
  assign(env, data_env)
}
# Now, DG2F_2017, G2FE_2017, G2LA_2017, G2FN_2018, G2FE_2018, G2LA_2018 contain the results for each environment

# Your list of data frames and harvest dates
harvest_dates <- c("DG2F_2017" = "7/25/2017", "G2FE_2017" = "7/31/2017", "G2LA_2017" = "8/10/2017", 
                   "G2FN_2018" = "7/28/2018", "G2FE_2018" = "8/4/2018", "G2LA_2018" = "8/16/2018")

process_data <- function(data) {
  melted_data <- melt(data[, 2:ncol(data)])
  transposed_data <- as.data.frame(t(melted_data$value))
  unique_vars <- t(melted_data$variable)
  num_weeks <- length(unique(data$Week))
  no_vars <- length(unique(unique_vars))
  weeks <- rep(data$Week[1:num_weeks], length(no_vars))
  colnames(transposed_data) <- paste0(unique_vars, " - Week ", weeks)
  return(transposed_data)
}

# Process DG2F_2017 data frame
transposed_DG2F_2017 <- process_data(DG2F_2017)

# Process G2FE_2017 data frame
transposed_G2FE_2017 <- process_data(G2FE_2017)

# Process G2LA_2017 data frame
transposed_G2LA_2017 <- process_data(G2LA_2017)

# Process G2FN_2018 data frame
transposed_G2FN_2018 <- process_data(G2FN_2018)

# Process G2FE_2018 data frame
transposed_G2FE_2018 <- process_data(G2FE_2018)

# Process G2LA_2018 data frame
transposed_G2LA_2018 <- process_data(G2LA_2018)

Field_Notes_2017 <- read.csv("CS17_G2F_FieldNotes.csv")
Field_Notes_2018 <- read.csv("CS18_FieldNotes_fulltable.csv")

# Step 1: Extracting specified columns and appending data
extracted_data_2017 <- Field_Notes_2017[, c("Barcode", "Year", "Test", "Stock", "Pedigree", "DTA", "DTS", "Moist", "TestWt", "Yield", "Population", "PlantDate", "Range", "Row", "HarvDate")]
extracted_data_2018 <- Field_Notes_2018[, c("Barcode", "Year", "Test", "Stock", "Pedigree", "DTA", "DTS", "Moist", "TestWt", "Yield", "Population", "PlantDate", "Range", "Row", "HarvDate")]
appended_data <- rbind(extracted_data_2017, extracted_data_2018)

# Step 2: Removing rows with empty cells and storing them in a new data frame for QC
rows_with_empty_cells <- appended_data[complete.cases(appended_data[ , !(names(appended_data) %in% c("PlantDate", "HarvDate"))]), ]
rows_with_empty_cells_qc <- appended_data[!complete.cases(appended_data[ , !(names(appended_data) %in% c("PlantDate", "HarvDate"))]), ]

# Create rows_with_empty_cells_qc if it doesn't exist
if (!exists("rows_with_empty_cells_qc")) {
  rows_with_empty_cells_qc <- data.frame()
}

# Identify rows with "LOCAL" in the Stock column
local_rows <- rows_with_empty_cells[grepl("LOCAL", rows_with_empty_cells$Stock), ]

# Append rows with "LOCAL" to rows_with_empty_cells_qc
rows_with_empty_cells_qc <- rbind(rows_with_empty_cells_qc, local_rows)

# Remove rows with "LOCAL" from rows_with_empty_cells
rows_with_empty_cells <- rows_with_empty_cells[!grepl("LOCAL", rows_with_empty_cells$Stock), ]

# Step 3: Create mappings between Test values, Year and Plant Date/Harvest Date
plant_dates <- c("DG2F_2017" = "3/3/2017", "G2FE_2017" = "3/3/2017", "G2LA_2017" = "4/6/2017", 
                 "G2FN_2018" = "3/7/2018", "G2FE_2018" = "3/8/2018", "G2LA_2018" = "4/3/2018")
harvest_dates <- c("DG2F_2017" = "7/25/2017", "G2FE_2017" = "7/31/2017", "G2LA_2017" = "8/10/2017", 
                   "G2FN_2018" = "7/28/2018", "G2FE_2018" = "8/4/2018", "G2LA_2018" = "8/16/2018")

#NOTE G2FN - 2018 DATE WAS APPROXIMATED AS NO DATE IN FIELD NOTES OR COMPETITION (G2FN_2018 = 7/28/2018).

# Create a new column that combines Test and Year
rows_with_empty_cells$TestYear <- paste(rows_with_empty_cells$Test, rows_with_empty_cells$Year, sep="_")

# Now use this new column for mapping
rows_with_empty_cells$PlantDate <- plant_dates[as.character(rows_with_empty_cells$TestYear)]
rows_with_empty_cells$HarvDate <- harvest_dates[as.character(rows_with_empty_cells$TestYear)]

# Step 4: subtract DTA/DTS dates
#Convert the PlantDate and HarvestDate column to Date type
rows_with_empty_cells$PlantDate <- as.Date(rows_with_empty_cells$PlantDate, format="%m/%d/%Y")
rows_with_empty_cells$HarvDate <- as.Date(rows_with_empty_cells$HarvDate, format="%m/%d/%Y")


# Check if DTA and DTS are already numeric (i.e., days difference already calculated)
if (!is.numeric(rows_with_empty_cells$DTA)) {
  # Only calculate for rows where DTA can be converted to a date
  rows_to_calculate <- !is.na(as.Date(rows_with_empty_cells$DTA, format="%m/%d/%Y"))
  
  rows_with_empty_cells$DTA[rows_to_calculate] <- as.numeric(as.Date(rows_with_empty_cells$DTA[rows_to_calculate], format="%m/%d/%Y") - rows_with_empty_cells$PlantDate[rows_to_calculate])
}

if (!is.numeric(rows_with_empty_cells$DTS)) {
  # Only calculate for rows where DTS can be converted to a date
  rows_to_calculate <- !is.na(as.Date(rows_with_empty_cells$DTS, format="%m/%d/%Y"))
  
  rows_with_empty_cells$DTS[rows_to_calculate] <- as.numeric(as.Date(rows_with_empty_cells$DTS[rows_to_calculate], format="%m/%d/%Y") - rows_with_empty_cells$PlantDate[rows_to_calculate])
}


# Step 5: Modifying cells in DTA and DTS columns that are above 70
rows_with_empty_cells$DTA[rows_with_empty_cells$DTA > 70] <- 70
rows_with_empty_cells$DTS[rows_with_empty_cells$DTS > 70] <- 70

#Convert the PlantDate and HarvestDate column to Date type
rows_with_empty_cells$PlantDate <- as.Date(rows_with_empty_cells$PlantDate, format="%m/%d/%Y")
rows_with_empty_cells$HarvDate <- as.Date(rows_with_empty_cells$HarvDate, format="%m/%d/%Y")

# Final merged data with adjusted dates
final_merged_data <- rows_with_empty_cells

# Create an empty dataframe with the same columns as final_merged_data
bound_df <- data.frame(matrix(ncol = ncol(final_merged_data) + ncol(transposed_DG2F_2017)))

# Set the column names for the new dataframe
colnames(bound_df) <- c(names(final_merged_data), names(transposed_DG2F_2017))

# Assign values from final_merged_data to the new dataframe
# Assuming columns 7 and 10 in final_merged_data are date columns
# Convert date values to strings and copy them to bound_df

bound_df[1:nrow(final_merged_data), 1:ncol(final_merged_data)] <- final_merged_data
bound_df[, 12] <- as.character(final_merged_data[, 12])
bound_df[, 15] <- as.character(final_merged_data[, 15])

# List of vectors corresponding to each label in TestYear
vectors <- list(DG2F_2017 = transposed_DG2F_2017,
                G2FE_2017 = transposed_G2FE_2017,
                G2LA_2017 = transposed_G2LA_2017,
                G2FN_2018 = transposed_G2FN_2018,
                G2FE_2018 = transposed_G2FE_2018,
                G2LA_2018 = transposed_G2LA_2018)

# Get the starting column index for appending data
start_column <- 17

# Iterate through the rows of bound_df and append vectors to columns starting from column 17
for (i in 1:nrow(bound_df)) {
  label <- bound_df$TestYear[i]
  if (label %in% names(vectors)) {
    vector <- vectors[[label]]
    bound_df[i, seq(start_column, start_column + length(vector) - 1)] <- vector
  }
}

# Split names based on "/"
split_pedigree <- strsplit(as.character(bound_df$Pedigree), "/")

# Create new columns Pedigree1 and Pedigree2
bound_df$Pedigree1 <- sapply(split_pedigree, function(x) ifelse(length(x) > 1, x[1], x))
bound_df$Pedigree2 <- sapply(split_pedigree, function(x) ifelse(length(x) > 1, x[2], x))

# Reorder the columns to move Pedigree1 and Pedigree2 to the start
bound_df <- bound_df[, c("Pedigree1", "Pedigree2", setdiff(names(bound_df), c("Pedigree1", "Pedigree2")))]

#Remove original Pedigree column.
bound_df <- subset(bound_df, select = -Pedigree)

#Julian Date by one start date.
# Convert character column to Date class
bound_df$PlantDate <- as.Date(bound_df$PlantDate, format = "%Y-%m-%d")

# Calculate Julian Date from the adjusted date column
bound_df$JulianPlantDateby1 <- as.numeric(bound_df$PlantDate) - as.numeric(as.Date("2017-01-01")) + 1

# Reorder columns with JulianPlantDateby1 as the first column
bound_df <- bound_df[, c("JulianPlantDateby1", names(bound_df)[!names(bound_df) %in% "JulianPlantDateby1"])]

#Julian Date per year.
# Calculate Julian Date by year
bound_df$JulianPlantDatePerYear <- ifelse(bound_df$Year == "2017",
                                   as.numeric(bound_df$PlantDate) - as.numeric(as.Date("2017-01-01")) + 1,
                                   as.numeric(bound_df$PlantDate) - as.numeric(as.Date("2018-01-01")) + 1)

# Reorder columns with JulianPlantDatePerYear as the first column
bound_df <- bound_df[, c("JulianPlantDatePerYear", names(bound_df)[!names(bound_df) %in% c("JulianPlantDatePerYear")])]


# Create a new column "TestYearRangeRow" by combining values
bound_df$TestYearRangeRow <- paste(bound_df$TestYear, bound_df$Range, bound_df$Row, sep = "_")

# Reorder columns with TestYearRangeRow as the first column
bound_df <- bound_df[, c("TestYearRangeRow", names(bound_df)[!names(bound_df) %in% "TestYearRangeRow"])]

remove_spaces <- function(df) {
  for (col in names(df)) {
    if (is.character(df[[col]])) {
      df[[col]] <- gsub("\\s", "", df[[col]])
    }
    # Note: You can add additional conditions for other data types if needed (e.g., numeric columns).
    # For numeric columns, you might want to convert them to character before removing spaces,
    # and then back to numeric after removing spaces, if necessary.
  }
  return(df)
}


bound_df <- remove_spaces(bound_df)


write.csv(bound_df, "Train_Val_Holdout_2017_2018.csv", row.names = TRUE, na = "")


##############################################################################
##############################2019  PROCESSING################################
##############################################################################

#NOTE I HAD TO PROCESS 2019 DIFFERENTLY, AS I INITIALLY USED IT AS HOLDOUT.
#MERGE THE TWO DATAFRAMES IF YOU WANT TO INCLUDE 2019 IN THE TRAINING/VALIDATION DATASET.

# Load necessary libraries
library(dplyr)
library(tidyr)
library(lubridate)
library(reshape2)


# Read in the data
data <- read.csv("2019_Data.csv")

# Extract the necessary columns
data_filtered <- data[, c("Env", "Date", "T2MDEW", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN", "PRECTOTCORR")]

# Convert the date column
data_filtered$Date <- as.Date(as.character(data_filtered$Date), format = "%Y%m%d")

# Define the initial dates and dataframe names
plant_dates <- c("DG2F_2019" = "2019-03-20", "G2F1_2019" = "2019-03-20", 
                 "G2LA_2019" = "2019-03-20")

# Create empty data frames for each environment
DG2F_2019 <- data.frame()
G2F1_2019 <- data.frame()
G2LA_2019 <- data.frame()


# Iterate through each environment and calculate the aggregations for every week
for (env in names(plant_dates)) {
  start_date <- as.Date(plant_dates[env])
  
  # Calculate week number from -13 to any weeks you want relative to the start date
  data_env <- data_filtered %>%
    filter(Date >= (start_date - weeks(13)), Date < (start_date + weeks(14))) %>% 
    mutate(Week = ((row_number() - 1) %/% 7) - 13) %>%
    group_by(Week) %>%
    summarise(T2MDEW = sum(T2MDEW),
              RH2M = mean(RH2M),
              WS2M = sum(WS2M),
              ALLSKY_SFC_SW_DWN = sum(ALLSKY_SFC_SW_DWN),
              T2M_MAX = mean(T2M_MAX),
              T2M_DIFF_SUM = sum(T2M_MAX - T2M_MIN),
              T2M_GDD = sum(((T2M_MAX + T2M_MIN)/2)-10),
              T2M_MIN = mean(T2M_MIN),
              PRECTOTCORR = sum(PRECTOTCORR)) %>%
    ungroup()
  # Assign the result to the corresponding data frame
  assign(env, data_env)
}

# Now, DG2F_2019 etc contain the results for each environment

# Your list of data frames and harvest dates
harvest_dates <- c("DG2F_2019" = "8/20/2019", "G2F1_2019" = "8/22/2019", "G2LA_2019" = "8/20/2019")

process_data <- function(data) {
  melted_data <- melt(data[, 2:ncol(data)])
  transposed_data <- as.data.frame(t(melted_data$value))
  unique_vars <- t(melted_data$variable)
  num_weeks <- length(unique(data$Week))
  no_vars <- length(unique(unique_vars))
  weeks <- rep(data$Week[1:num_weeks], length(no_vars))
  colnames(transposed_data) <- paste0(unique_vars, " - Week ", weeks)
  return(transposed_data)
}

# Process DG2F_2019 data frame
transposed_DG2F_2019 <- process_data(DG2F_2019)

# Process G2F1_2019 data frame
transposed_G2F1_2019 <- process_data(G2F1_2019)

# Process G2LA_2019 data frame
transposed_G2LA_2019 <- process_data(G2LA_2019)



# Read the dataframes
Field_Notes_2019 <- read.csv("CS19_FieldNotes_fulltable.csv")
Field_Notes_2019_DG2F <- read.csv("CS19_FieldNotes_DG2F.csv")
Field_Notes_2019_G2F1 <- read.csv("CS19_FieldNotes_G2F1.csv")
Field_Notes_2019_G2LA <- read.csv("CS19_FieldNotes_G2LA.csv")

# Assuming the dataframe is named "Field_Notes_2019"
# Duplicate every unique row and add -1 and -2 suffixes
Field_Notes_2019 <- Field_Notes_2019[rep(seq_len(nrow(Field_Notes_2019)), each = 2), ]
Field_Notes_2019$Barcode <- paste0(Field_Notes_2019$Barcode, c("-1", "-2"))

# Assuming the dataframe is named "Field_Notes_2019_DG2F"
# Duplicate every unique row and add -1 and -2 suffixes
Field_Notes_2019_DG2F <- Field_Notes_2019_DG2F[rep(seq_len(nrow(Field_Notes_2019_DG2F)), each = 2), ]
Field_Notes_2019_DG2F$Barcode <- paste0(Field_Notes_2019_DG2F$Barcode, c("-1", "-2"))

# Assuming the dataframe is named "Field_Notes_2019_G2LA"
# Duplicate every unique row and add -1 and -2 suffixes
Field_Notes_2019_G2LA <- Field_Notes_2019_G2LA[rep(seq_len(nrow(Field_Notes_2019_G2LA)), each = 2), ]
Field_Notes_2019_G2LA$Barcode <- paste0(Field_Notes_2019_G2LA$Barcode, c("-1", "-2"))


# Duplicate and rename original Range and Row columns
Field_Notes_2019$Range_Old <- Field_Notes_2019$Range
Field_Notes_2019$Row_Old <- Field_Notes_2019$Row

# Assuming the column is called "Plot"
# Remove rows with the value of 0 in the "Plot" column
Field_Notes_2019_G2F1 <- Field_Notes_2019_G2F1[Field_Notes_2019_G2F1$Plot != 0, ]


# Merge dataframes to update Field_Notes_2019
Field_Notes_2019 <- merge(Field_Notes_2019, Field_Notes_2019_DG2F[, c("Barcode", "Range", "Row")], by = "Barcode", all.x = TRUE)
names(Field_Notes_2019)[names(Field_Notes_2019) == "Range.y"] <- "Range_DG2F"
names(Field_Notes_2019)[names(Field_Notes_2019) == "Row.y"] <- "Row_DG2F"

Field_Notes_2019 <- merge(Field_Notes_2019, Field_Notes_2019_G2F1[, c("Barcode", "Range", "Row")], by = "Barcode", all.x = TRUE)
names(Field_Notes_2019)[names(Field_Notes_2019) == "Range"] <- "Range_G2F1"
names(Field_Notes_2019)[names(Field_Notes_2019) == "Row"] <- "Row_G2F1"

Field_Notes_2019 <- merge(Field_Notes_2019, Field_Notes_2019_G2LA[, c("Barcode", "Range", "Row")], by = "Barcode", all.x = TRUE)
names(Field_Notes_2019)[names(Field_Notes_2019) == "Range"] <- "Range_G2LA"
names(Field_Notes_2019)[names(Field_Notes_2019) == "Row"] <- "Row_G2LA"


# Create new columns Range_New and Row_New based on conditions
Field_Notes_2019$Range <- ifelse(!is.na(Field_Notes_2019$Range_DG2F), Field_Notes_2019$Range_DG2F,
                                     ifelse(!is.na(Field_Notes_2019$Range_G2F1), Field_Notes_2019$Range_G2F1,
                                            Field_Notes_2019$Range_G2LA))
Field_Notes_2019$Row <- ifelse(!is.na(Field_Notes_2019$Row_DG2F), Field_Notes_2019$Row_DG2F,
                                   ifelse(!is.na(Field_Notes_2019$Row_G2F1), Field_Notes_2019$Row_G2F1,
                                          Field_Notes_2019$Row_G2LA))

# Remove unnecessary columns
Field_Notes_2019 <- Field_Notes_2019[, !(names(Field_Notes_2019) %in% c("Range_DG2F", "Row_DG2F", "Range_G2F1", "Row_G2F1", "Range_G2LA", "Row_G2LA"))]

# Remove rows with NA in the Row_New column
# Remove rows with empty cells in the Row_New column
Field_Notes_2019 <- Field_Notes_2019[!(Field_Notes_2019$Row %in% c("", NA)), ]


# Step 1: Extracting specified columns and appending data
appended_data <- Field_Notes_2019[, c("Barcode", "Year", "Test", "Stock", "Pedigree", "DTA", "DTS", "Moist", "TestWt", "Yield", "Population", "PlantDate", "Range", "Row", "HarvDate")]

# Step 2: Removing rows with empty cells and storing them in a new data frame for QC
#modified from previous code as Stock column for some reason processed some rows with empty cells or "NA"..
rows_with_empty_cells <- appended_data[complete.cases(appended_data[, !(names(appended_data) %in% c("PlantDate", "HarvDate"))]) & !(appended_data$Stock == "" | appended_data$Stock == " NA" | appended_data$Stock == "NA ") , ]
rows_with_empty_cells_qc <- appended_data[!complete.cases(appended_data[, !(names(appended_data) %in% c("PlantDate", "HarvDate"))]) | (appended_data$Stock == "" | appended_data$Stock == " NA" | appended_data$Stock == "NA "), ]


# Create rows_with_empty_cells_qc if it doesn't exist
if (!exists("rows_with_empty_cells_qc")) {
  rows_with_empty_cells_qc <- data.frame()
}

# Identify rows with "LOCAL" in the Stock column
local_rows <- rows_with_empty_cells[grepl("LOCAL", rows_with_empty_cells$Stock), ]

# Append rows with "LOCAL" to rows_with_empty_cells_qc
rows_with_empty_cells_qc <- rbind(rows_with_empty_cells_qc, local_rows)

# Remove rows with "LOCAL" from rows_with_empty_cells
rows_with_empty_cells <- rows_with_empty_cells[!grepl("LOCAL", rows_with_empty_cells$Stock), ]

# Step 3: Create mappings between Test values, Year and Plant Date/Harvest Date
plant_dates <- c("DG2F_2019" = "3/20/2019", "G2F1_2019" = "3/20/2019", "G2LA_2019" = "3/20/2019")
harvest_dates <- c("DG2F_2019" = "8/20/2019", "G2F1_2019" = "8/22/2019", "G2LA_2019" = "8/20/2019")

# Create a new column that combines Test and Year
rows_with_empty_cells$TestYear <- paste(rows_with_empty_cells$Test, rows_with_empty_cells$Year, sep="_")

# Now use this new column for mapping
rows_with_empty_cells$PlantDate <- plant_dates[as.character(rows_with_empty_cells$TestYear)]
rows_with_empty_cells$HarvDate <- harvest_dates[as.character(rows_with_empty_cells$TestYear)]


# Step 4: subtract DTA/DTS dates
#Convert the PlantDate and HarvestDate column to Date type
rows_with_empty_cells$PlantDate <- as.Date(rows_with_empty_cells$PlantDate, format="%m/%d/%Y")
rows_with_empty_cells$HarvDate <- as.Date(rows_with_empty_cells$HarvDate, format="%m/%d/%Y")


# Check if DTA and DTS are already numeric (i.e., days difference already calculated)
if (!is.numeric(rows_with_empty_cells$DTA)) {
  # Only calculate for rows where DTA can be converted to a date
  rows_to_calculate <- !is.na(as.Date(rows_with_empty_cells$DTA, format="%m/%d/%Y"))
  
  rows_with_empty_cells$DTA[rows_to_calculate] <- as.numeric(as.Date(rows_with_empty_cells$DTA[rows_to_calculate], format="%m/%d/%Y") - rows_with_empty_cells$PlantDate[rows_to_calculate])
}

if (!is.numeric(rows_with_empty_cells$DTS)) {
  # Only calculate for rows where DTS can be converted to a date
  rows_to_calculate <- !is.na(as.Date(rows_with_empty_cells$DTS, format="%m/%d/%Y"))
  
  rows_with_empty_cells$DTS[rows_to_calculate] <- as.numeric(as.Date(rows_with_empty_cells$DTS[rows_to_calculate], format="%m/%d/%Y") - rows_with_empty_cells$PlantDate[rows_to_calculate])
}


# Step 5: Modifying cells in DTA and DTS columns that are above 70
rows_with_empty_cells$DTA[rows_with_empty_cells$DTA > 70] <- 70
rows_with_empty_cells$DTS[rows_with_empty_cells$DTS > 70] <- 70

#Convert the PlantDate and HarvestDate column to Date type
rows_with_empty_cells$PlantDate <- as.Date(rows_with_empty_cells$PlantDate, format="%m/%d/%Y")
rows_with_empty_cells$HarvDate <- as.Date(rows_with_empty_cells$HarvDate, format="%m/%d/%Y")

# Final merged data with adjusted dates
final_merged_data <- rows_with_empty_cells

# Create an empty dataframe with the same columns as final_merged_data
bound_df <- data.frame(matrix(ncol = ncol(final_merged_data) + ncol(transposed_DG2F_2019)))

# Set the column names for the new dataframe
colnames(bound_df) <- c(names(final_merged_data), names(transposed_DG2F_2019))

# Assign values from final_merged_data to the new dataframe
# Assuming columns 7 and 10 in final_merged_data are date columns
# Convert date values to strings and copy them to bound_df

bound_df[1:nrow(final_merged_data), 1:ncol(final_merged_data)] <- final_merged_data
bound_df[, 12] <- as.character(final_merged_data[, 12])
bound_df[, 15] <- as.character(final_merged_data[, 15])

# List of vectors corresponding to each label in TestYear
vectors <- list(DG2F_2019 = transposed_DG2F_2019,
                G2F1_2019 = transposed_G2F1_2019,
                G2LA_2019 = transposed_G2LA_2019)

# Get the starting column index for appending data
start_column <- 17

# Iterate through the rows of bound_df and append vectors to columns starting from column 17
for (i in 1:nrow(bound_df)) {
  label <- bound_df$TestYear[i]
  if (label %in% names(vectors)) {
    vector <- vectors[[label]]
    bound_df[i, seq(start_column, start_column + length(vector) - 1)] <- vector
  }
}

# Split names based on "/"
split_pedigree <- strsplit(as.character(bound_df$Pedigree), "/")

# Create new columns Pedigree1 and Pedigree2
bound_df$Pedigree1 <- sapply(split_pedigree, function(x) ifelse(length(x) > 1, x[1], x))
bound_df$Pedigree2 <- sapply(split_pedigree, function(x) ifelse(length(x) > 1, x[2], x))

# Reorder the columns to move Pedigree1 and Pedigree2 to the start
bound_df <- bound_df[, c("Pedigree1", "Pedigree2", setdiff(names(bound_df), c("Pedigree1", "Pedigree2")))]

#Remove original Pedigree column.
bound_df <- subset(bound_df, select = -Pedigree)

#Julian Date by one start date.
# Convert character column to Date class
bound_df$PlantDate <- as.Date(bound_df$PlantDate, format = "%Y-%m-%d")

# Calculate Julian Date from the adjusted date column
bound_df$JulianPlantDateby1 <- as.numeric(bound_df$PlantDate) - as.numeric(as.Date("2019-01-01")) + 1

# Reorder columns with JulianPlantDateby1 as the first column
bound_df <- bound_df[, c("JulianPlantDateby1", names(bound_df)[!names(bound_df) %in% "JulianPlantDateby1"])]

#Julian Date per year.
# Calculate Julian Date by year
bound_df$JulianPlantDatePerYear <- ifelse(bound_df$Year == "2019",
                                          as.numeric(bound_df$PlantDate) - as.numeric(as.Date("2019-01-01")) + 1)

# Reorder columns with JulianPlantDatePerYear as the first column
bound_df <- bound_df[, c("JulianPlantDatePerYear", names(bound_df)[!names(bound_df) %in% c("JulianPlantDatePerYear")])]


# Create a new column "TestYearRangeRow" by combining values
bound_df$TestYearRangeRow <- paste(bound_df$TestYear, bound_df$Range, bound_df$Row, sep = "_")

# Reorder columns with TestYearRangeRow as the first column
bound_df <- bound_df[, c("TestYearRangeRow", names(bound_df)[!names(bound_df) %in% "TestYearRangeRow"])]

remove_spaces <- function(df) {
  for (col in names(df)) {
    if (is.character(df[[col]])) {
      df[[col]] <- gsub("\\s", "", df[[col]])
    }
    # Note: You can add additional conditions for other data types if needed (e.g., numeric columns).
    # For numeric columns, you might want to convert them to character before removing spaces,
    # and then back to numeric after removing spaces, if necessary.
  }
  return(df)
}


bound_df <- remove_spaces(bound_df)

# Duplicate and rename the column
bound_df$Replicate <- bound_df$Barcode

# Reorder columns with the duplicated column as the first one
bound_df <- bound_df[c("Replicate", names(bound_df)[-length(bound_df)])]

# Duplicate and rename the column
bound_df$ID <- bound_df$Barcode

# Reorder columns with the duplicated column as the first one
bound_df <- bound_df[c("ID", names(bound_df)[-length(bound_df)])]

bound_df$Barcode <- substring(bound_df$Barcode, 1, nchar(bound_df$Barcode) - 2)

write.csv(bound_df, "2019_Training_Val_Holdout.csv", row.names = FALSE, na = "")


