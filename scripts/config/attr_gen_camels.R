#' @title Generate attributes for CAMELS basins
#' @description This script uses the proc.attr.hydfab package to acquire attributes
#' of interest.
#'


library(dplyr)
library(glue)
library(tidyr)
library(yaml)
library(proc.attr.hydfab)

main <- function(){
  # Define args supplied to command line
  home_dir <- Sys.getenv("HOME")

  ############################ BEGIN CUSTOM MUNGING ############################

  # ----------------------=-- Read in CAMELS gage ids ------------------------ #
  path_gages_ii <- glue::glue("{home_dir}/noaa/camels/gagesII_wood/gages_list.txt")
  dat_gages_ii <- read.csv(path_gages_ii)
  gage_ids <- base::lapply(1:nrow(dat_gages_ii), function(i)
    tail(strsplit(dat_gages_ii[i,],split = ' ',fixed = TRUE)[[1]],n=1)) |>
    unlist() |>
    lapply(function(x)
    gsub(pattern=".gpkg",replacement = "",x = x)) |>
    unlist() |>
    lapply( function(x) gsub(pattern = "Gage_", replacement = "",x=x)) |>
    unlist()

  utils::write.table(gage_ids,glue::glue('{home_dir}/noaa/camels/gagesII_wood/camels_ii_gage_ids.txt'),row.names = FALSE,col.names = FALSE)

  # --------------------- Read in usgs NHD attribute IDs --------------------- #
  # Read desired usgs nhdplus attributes, stored in NOAA shared drive here:
  # https://docs.google.com/spreadsheets/d/1h-630L2ChH5zlQIcWJHVaxY9YXtGowcCqakQEAXgRrY/edit?usp=sharing
  attrs_nhd_df <- read.csv(glue::glue("{home_dir}/noaa/regionalization/processing/usgs_nhdplus_attrs.csv"))

  attrs_nhd <-   attrs_nhd_df$ID

  Retr_Params <- list(paths = list(dir_db_attrs = glue::glue("{home_dir}/noaa/regionalization/data/input/attributes/"),
                                   dir_std_base = glue::glue("{home_dir}/noaa/regionalization/data/input/user_data_std")),
                      vars = list(usgs_vars = attrs_nhd),
                      datasets = "camelsii_nhdp_grab_nov24",
                      xtra_hfab = list(hfab_retr=FALSE))


  ############################ END CUSTOM MUNGING ##############################

  # ---------------------- Grab all needed attributes ---------------------- #
  # Now acquire the attributes:
  ls_comids <- proc.attr.hydfab::proc_attr_gageids(gage_ids=gage_ids,
                                                   featureSource='nwissite',
                                                   featureID='USGS-{gage_id}',
                                                   Retr_Params=Retr_Params,
                                                   overwrite=FALSE)

  message(glue::glue("Completed attribute acquisition for {Retr_Params$paths$dir_db_attrs}"))
}


main()
