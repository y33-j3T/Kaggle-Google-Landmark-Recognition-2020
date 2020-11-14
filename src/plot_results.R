library(tidyverse)

m <- c("ResNet50 v2", "Xception", "VGG16")
da <- factor(c("With data augmentation", "Without data augmentation"), levels = c("Without data augmentation", "With data augmentation"))
ft <- factor(c("Non-Fine-tuned", "Fine-tuned"), levels = c("Non-Fine-tuned", "Fine-tuned"))
data.frame(
  model = rep(m, each = 2, 2),
  data_augmented = rep(da, each = 1, 6),
  fine_tuned = rep(ft, each=6),
  MAP = c(0.95, 0.94, 0.93, 0.95, 0.89, 0.92, 0.95, 0.94, 0.95, 0.94, 0.96, 0.93)
) %>% 
  # filter(fine_tuned == "Fine-tuned") %>% 
  ggplot(aes(x = data_augmented, y = MAP, color = model)) +
  geom_line(aes(group = model), position=position_jitter(w=0.03, h=0)) +
  geom_hline(yintercept=0.87) +
  geom_vline(xintercept=0.5) +
  facet_wrap( ~ fine_tuned, ncol = 2) +
  theme_bw() +
  theme(
    legend.title = element_blank(),
    axis.title.x = element_blank(),
    axis.text.x = element_text(size = 12),
    panel.border = element_blank(),
    strip.text = element_text(size = 12),
    strip.background=element_rect(colour = "white", fill = "white"),
    legend.text = element_text(size = 12)
  )
