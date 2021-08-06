library(covid19.nhs.data)

adm <- get_admissions("ltla")
#map_admissions(adm, england_ltla_shape)

library(ggplot2)
library(dplyr)
#> 
#> Attaching package: 'dplyr'
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union

adm %>% 
  filter(geo_name %in% "Derby") %>% 
  ggplot(aes(x = date, y = admissions)) + 
  geom_col(width = 0.9, col = "grey50", fill = "grey85") +
  theme_minimal() +
  labs(x = "Date", y = "Daily Hospital Admissions",
       title = "Covid-19 Admissions in Derby", 
       subtitle = "Estimated using a probabilistic mapping from NHS Trusts to lower-tier local authority level")

# download_trust_data()

adm_utla = get_admissions(level = "utla")

adm_ltla = get_admissions(level = "ltla")

write.csv(adm_utla,'2021_08_04-admissions_utla.csv')

tt = trust_utla_mapping

write.csv(trust_utla_mapping,'trust_utla_mapping.csv')