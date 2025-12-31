# Load libraries
library(ggplot2)
library(dplyr)
library(readr)
library(scales)
library(viridis)
 
library(stringr)
library(tidyr)
library(purrr)
library(rlang)  # for tidy-eval (.data, sym)
library(utils)

setwd("~/Desktop/Results")

 

# -----------------------------------------------------------------------------
# Integrated MAG analysis (CheckM1/2 + GTDB-Tk + metaWRAP [+ optional RefineM])
# Uses color palettes consistent with maps.R (citystat_core12)
# Outputs: unified table + multiple figures
# -----------------------------------------------------------------------------

# Palettes carried over from maps.R
citystat_core12 <- c(
  Navy      = "#233B5D",
  Blue      = "#2F6DA8",
  DeepTeal  = "#1C7C7D",
  Cyan      = "#17BEBB",
  Forest    = "#2E7D32",
  Olive     = "#7A8F30",
  Mustard   = "#C7A41A",
  Orange    = "#E07A2D",
  Vermilion = "#D0432B",
  Crimson   = "#B01E2F",
  Plum      = "#7A3E9D",
  Indigo    = "#3B4BA3"
)

# Create output dir for this section
dir.create("mag_analysis", showWarnings = FALSE)

# Helpers ----------------------------------------------------------------------

norm_mag_id <- function(x) {
  # Drop trailing dots sometimes present in CheckM2 names
  x <- sub("\\.$", "", x)
  x
}

parse_tax <- function(classification) {
  # Split GTDB taxonomy string into ranks
  parts <- strsplit(classification %||% "", ";")[[1]]
  parts <- trimws(parts)
  get_rank <- function(prefix) {
    hit <- parts[startsWith(parts, prefix)]
    if (length(hit) == 0) return(NA_character_)
    sub(prefix, "", hit[1], fixed = TRUE)
  }
  tibble(
    domain = get_rank("d__"),
    phylum = get_rank("p__"),
    class = get_rank("c__"),
    order = get_rank("o__"),
    family = get_rank("f__"),
    genus = get_rank("g__"),
    species = get_rank("s__")
  )
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

# Loaders ----------------------------------------------------------------------

load_checkm1_all <- function(path = "checkm1_out/bin_stats_ext.tsv") {
  if (!file.exists(path)) return(tibble())
  df <- read_tsv(path, comment = "[", show_col_types = FALSE)
  # Safely capture marker count columns if available
  has_markers <- "# markers" %in% names(df)
  has_marker_sets <- "# marker sets" %in% names(df)
  df %>%
    mutate(
      Markers_chk1 = if (has_markers) as.numeric(.data[["# markers"]]) else NA_real_,
      MarkerSets_chk1 = if (has_marker_sets) as.numeric(.data[["# marker sets"]]) else NA_real_
    ) %>%
    transmute(
      mag_id = .data[["Bin Id"]] |> norm_mag_id(),
      Completeness_chk1 = as.numeric(.data$Completeness),
      Contamination_chk1 = as.numeric(.data$Contamination),
      Strain_heterogeneity = as.numeric(.data[["Strain heterogeneity"]]),
      Contigs_chk1 = as.numeric(.data[["# contigs"]]),
      Genome_size_chk1 = as.numeric(.data[["Genome size (bp)"]]),
      N50_chk1 = as.numeric(.data[["N50 (contigs)"]]),
      GC_chk1 = as.numeric(.data$GC)/100,
      Markers_chk1 = .data$Markers_chk1,
      MarkerSets_chk1 = .data$MarkerSets_chk1
    ) %>%
    mutate(Quality1 = .data$Completeness_chk1 - 5*.data$Contamination_chk1)
}

load_checkm2_all <- function(path = "checkm2_out/quality_report.tsv") {
  if (!file.exists(path)) return(tibble())
  read_tsv(path, show_col_types = FALSE) %>%
    transmute(
  mag_id = .data$Name |> norm_mag_id(),
  Completeness_chk2 = as.numeric(.data$Completeness),
  Contamination_chk2 = as.numeric(.data$Contamination),
  Genome_Size = as.numeric(.data$Genome_Size),
  Total_Contigs = as.numeric(.data$Total_Contigs),
  Contig_N50 = as.numeric(.data$Contig_N50),
  GC_Content = as.numeric(.data$GC_Content),
      Coding_Density = as.numeric(.data$Coding_Density)
    ) %>%
    mutate(Quality2 = .data$Completeness_chk2 - 5*.data$Contamination_chk2)
}

find_files <- function(root, pattern) {
  paths <- list.dirs(root, recursive = TRUE, full.names = TRUE)
  files <- file.path(paths, pattern)
  files[file.exists(files)]
}

load_gtdb_all <- function(root = "gtdbtk_drep_out") {
  if (!dir.exists(root)) return(tibble())
  # Prefer bac120; fall back to ar53 if needed
  bac_files <- find_files(root, file.path("classify", "gtdbtk.bac120.summary.tsv"))
  ar_files  <- find_files(root, file.path("classify", "gtdbtk.ar53.summary.tsv"))
  files <- c(bac_files, ar_files)
  if (length(files) == 0) return(tibble())
  map_dfr(files, function(fp) {
    df <- suppressWarnings(read_tsv(fp, show_col_types = FALSE))
    if (!all(c("user_genome", "classification") %in% names(df))) return(tibble())
    out <- df %>%
  transmute(mag_id = norm_mag_id(.data$user_genome), classification = .data$classification)
    # expand ranks
  ranks <- map_dfr(out$classification, parse_tax)
  bind_cols(out %>% select(all_of(c("mag_id", "classification"))), ranks) %>%
      mutate(sample = basename(dirname(dirname(fp))))
  }) %>%
    group_by(.data$mag_id) %>%
    slice_head(n = 1) %>%
    ungroup()
}

load_metawrap_all <- function(root = "metawrap_stats_only") {
  if (!dir.exists(root)) return(tibble())
  stat_paths <- list.files(root, pattern = "^SAMPLE-"  # adjust for your sample folder naming, full.names = TRUE)
  stat_paths <- stat_paths[file.info(stat_paths)$isdir]
  map_dfr(stat_paths, function(sp) {
    stats_file <- file.path(sp, "metawrap_50_10_bins.stats")
    if (!file.exists(stats_file)) return(tibble())
    smp <- basename(sp)
    read_tsv(stats_file, show_col_types = FALSE) %>%
      mutate(
        sample = smp,
  mag_id = paste0(smp, "_", .data$bin, ".filtered") |> norm_mag_id()
      ) %>%
      transmute(
        mag_id = .data$mag_id,
        sample = .data$sample,
        meta_completeness = as.numeric(.data$completeness),
        meta_contamination = as.numeric(.data$contamination),
        meta_GC = as.numeric(.data$GC),
        meta_N50 = as.numeric(.data$N50),
        meta_size = as.numeric(.data$size),
        meta_binner = .data$binner,
        meta_lineage = .data$lineage
      )
  })
}

# Optional: compute per-MAG mean coverage from RefineM outputs if available
# This assumes a directory layout refinem_stats/<sample>/{coverage.tsv, scaffold_stats.tsv}
# and that scaffold_stats.tsv contains a 'Bin Id' or 'bin' column with mag_id-compatible names
load_refinem_coverage <- function(root = "refinem_stats") {
  if (!dir.exists(root)) return(tibble())
  smps <- list.files(root, pattern = "^SAMPLE-"  # adjust for your sample folder naming, full.names = TRUE)
  smps <- smps[file.info(smps)$isdir]
  map_dfr(smps, function(sp) {
    cov_fp <- file.path(sp, "coverage.tsv")
    scaf_fp <- file.path(sp, "scaffold_stats.tsv")
    if (!file.exists(cov_fp) || !file.exists(scaf_fp)) return(tibble())
    # Read minimal columns to reduce memory; rely on data.table::fread if present
    cov <- tryCatch(read_tsv(cov_fp, show_col_types = FALSE), error = function(e) NULL)
    scf <- tryCatch(suppressMessages(read_tsv(scaf_fp, show_col_types = FALSE)), error = function(e) NULL)
    if (is.null(cov) || is.null(scf)) return(tibble())
    # Heuristics: expect columns like 'Scaffold Id', sample column for coverage, and a bin column
    cov_nm <- names(cov)
    samp_col <- cov_nm[grepl(basename(sp), cov_nm, fixed = TRUE)] %||% cov_nm[length(cov_nm)]
    scf_bin_col <- names(scf)[tolower(names(scf)) %in% c("bin id", "bin", "mag_id")]
    scf_scaf_col <- names(scf)[tolower(names(scf)) %in% c("scaffold id", "scaffold", "contig", "sequence")]
    if (length(scf_bin_col) == 0 || length(scf_scaf_col) == 0) return(tibble())
    scf_min <- scf %>% select(scaffold = !!sym(scf_scaf_col[1]), bin = !!sym(scf_bin_col[1]))
    cov_min <- cov %>% select(scaffold = 1, coverage = !!sym(samp_col))
    cov_agg <- cov_min %>%
      inner_join(scf_min, by = "scaffold") %>%
      mutate(mag_id = norm_mag_id(.data$bin)) %>%
      group_by(.data$mag_id) %>%
      summarize(refinem_mean_cov = mean(as.numeric(.data$coverage), na.rm = TRUE), .groups = "drop")
    cov_agg
  })
}

# Build unified table ----------------------------------------------------------------

chk1 <- load_checkm1_all()
chk2 <- load_checkm2_all()
gtdb <- load_gtdb_all()
mw   <- load_metawrap_all()
refc <- tryCatch(load_refinem_coverage(), error = function(e) tibble())
if (!("mag_id" %in% names(refc))) {
  refc <- tibble(mag_id = character(), refinem_mean_cov = numeric())
}

unified <- chk1 %>%
  full_join(chk2, by = "mag_id") %>%
  # Attach GTDB taxonomy
  left_join(gtdb %>% select(mag_id, classification, domain, phylum, class, order, family, genus, species), by = "mag_id") %>%
  # Attach metaWRAP stats (sample also comes from here when present)
  left_join(mw, by = c("mag_id" = "mag_id")) %>%
  # Attach optional RefineM coverage
  left_join(refc, by = "mag_id") %>%
  mutate(
    sample = coalesce(sample, str_extract(mag_id, "[A-Za-z0-9-]+_S\\d+")),
    mimag_chk1 = !is.na(Completeness_chk1) & Completeness_chk1 >= 90 & coalesce(Contamination_chk1, 1000) <= 5,
    mimag_chk2 = !is.na(Completeness_chk2) & Completeness_chk2 >= 90 & coalesce(Contamination_chk2, 1000) <= 5,
    classification_mimag = case_when(
      mimag_chk1 & mimag_chk2 ~ "High-quality MAG",
      mimag_chk1 | mimag_chk2 ~ "Medium-quality MAG",
      TRUE ~ "Low-quality MAG"
    ),
    Quality1 = ifelse(is.na(Quality1) & !is.na(Completeness_chk1) & !is.na(Contamination_chk1), Completeness_chk1 - 5*Contamination_chk1, Quality1),
    Quality2 = ifelse(is.na(Quality2) & !is.na(Completeness_chk2) & !is.na(Contamination_chk2), Completeness_chk2 - 5*Contamination_chk2, Quality2),
    quadrant = case_when(
      coalesce(Quality1, -Inf) >= 65 & coalesce(Quality2, -Inf) >= 65 ~ "High Quality (Both)",
      coalesce(Quality1, -Inf) >= 65 & coalesce(Quality2, -Inf) <  65 ~ "CheckM1 Only",
      coalesce(Quality1, -Inf) <  65 & coalesce(Quality2, -Inf) >= 65 ~ "CheckM2 Only",
      TRUE ~ "Low Quality (Both)"
    )
  )

# Save table
write_tsv(unified, file.path("mag_analysis", "MAG_quality_summary.tsv"))

# Figures ----------------------------------------------------------------------

# Classification colors using citystat_core12
classification_colors <- c(
  "High-quality MAG"   = citystat_core12[["Blue"]],
  "Medium-quality MAG" = citystat_core12[["Orange"]],
  "Low-quality MAG"    = citystat_core12[["Crimson"]]
)

quad_colors2 <- c(
  "High Quality (Both)" = citystat_core12[["Cyan"]],
  "CheckM1 Only"        = citystat_core12[["DeepTeal"]],
  "CheckM2 Only"        = citystat_core12[["Plum"]],
  "Low Quality (Both)"  = citystat_core12[["Vermilion"]]
)

# 1) Quality scatter (updated palette)
p_scatter <- ggplot(unified, aes(Quality1, Quality2)) +
  annotate("rect", xmin=65, xmax=105, ymin=65, ymax=105, fill=alpha(quad_colors2["High Quality (Both)"], 0.12))+
  annotate("rect", xmin=-5, xmax=65, ymin=65, ymax=105, fill=alpha(quad_colors2["CheckM2 Only"], 0.12))+
  annotate("rect", xmin=65, xmax=105, ymin=-5, ymax=65, fill=alpha(quad_colors2["CheckM1 Only"], 0.12))+
  annotate("rect", xmin=-5, xmax=65, ymin=-5, ymax=65, fill=alpha(quad_colors2["Low Quality (Both)"], 0.12))+
  geom_vline(xintercept=65, linetype="dashed", color="grey40")+
  geom_hline(yintercept=65, linetype="dashed", color="grey40")+
  geom_point(aes(color = classification_mimag, size = Genome_Size), alpha = 0.75, stroke = 0.4) +
  scale_color_manual(values = classification_colors, name = "MIMAG")+
  scale_size_continuous(range = c(0.8, 3), labels = label_number(scale_cut = cut_short_scale()), name = "Genome Size")+
  scale_x_continuous(limits=c(-5,105))+
  scale_y_continuous(limits=c(-5,105))+
  labs(x = "CheckM1 Quality (Completeness - 5xContam)", y = "CheckM2 Quality (Completeness - 5xContam)",
       title = "MAG Quality: CheckM1 vs CheckM2 (maps palette)")+
  theme_minimal(base_size = 12) + theme(legend.position = "right")

ggsave(file.path("mag_analysis", "mag_quality_scatter.png"), p_scatter, width=10, height=8, dpi=300, bg="white")

# 2) Per-sample counts by MIMAG class
per_sample_counts <- unified %>% filter(!is.na(sample)) %>%
  count(sample, classification_mimag) %>%
  group_by(sample) %>% mutate(prop = n/sum(n)) %>% ungroup()

# Reorder samples by total MAG count (descending)
sample_order <- per_sample_counts %>% group_by(sample) %>% summarize(total = sum(n), .groups = "drop") %>% arrange(desc(total)) %>% pull(sample)
per_sample_counts <- per_sample_counts %>% mutate(sample = factor(sample, levels = sample_order))

p_stack <- ggplot(per_sample_counts, aes(x = sample, y = n, fill = classification_mimag)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = classification_colors, name = "MIMAG") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1)) +
  labs(title = "MAG counts per sample by MIMAG class", x = "Sample", y = "Count")

ggsave(file.path("mag_analysis", "mag_per_sample_quality.png"), p_stack, width=12, height=6, dpi=300, bg="white")

# 3) Static sunburst taxonomy of High-quality MAGs (domain → species)
{
  # Prepare leaf counts with paths truncated at deepest classified rank; drop MAGs lacking phylum entirely
  ranks <- c("domain", "phylum", "class", "order", "family", "genus", "species")
  leaves <- unified %>%
    filter(classification_mimag == "High-quality MAG") %>%
    transmute(
      domain = ifelse(is.na(domain) | domain == "", NA_character_, domain),
      phylum = ifelse(is.na(phylum) | phylum == "", NA_character_, phylum),
      class  = ifelse(is.na(class)  | class  == "", NA_character_, class),
      order  = ifelse(is.na(order)  | order  == "", NA_character_, order),
      family = ifelse(is.na(family) | family == "", NA_character_, family),
      genus  = ifelse(is.na(genus)  | genus  == "", NA_character_, genus),
      species= ifelse(is.na(species)| species== "", NA_character_, species)
    ) %>%
    filter(!is.na(phylum))

  if (nrow(leaves) > 0) {
    # Collapse to deepest non-NA path per MAG
    path_str <- apply(leaves, 1, function(row) {
      nz <- row[!is.na(row)]
      paste(nz, collapse = "|")
    })
    leaf_counts <- tibble(path = path_str) %>% count(path, name = "value")

  # Build nodes table: accumulate counts for each prefix
    build_nodes <- function(paths, values) {
      nodes <- list()
      for (i in seq_along(paths)) {
    # Split on literal '|' (use regex split, not fixed, so "\\|" works)
    parts <- strsplit(paths[i], "\\|", fixed = FALSE)[[1]]
        for (d in seq_along(parts)) {
          id <- paste(parts[seq_len(d)], collapse = "|")
          parent <- if (d == 1) "" else paste(parts[seq_len(d-1)], collapse = "|")
          key <- id
          if (is.null(nodes[[key]])) {
            nodes[[key]] <- list(id = id, parent = parent, label = parts[d], depth = d, value = 0)
          }
          nodes[[key]]$value <- nodes[[key]]$value + values[i]
        }
      }
      as_tibble(do.call(rbind, lapply(nodes, as.data.frame)))
    }

    nodes <- build_nodes(leaf_counts$path, leaf_counts$value)
    # Recompute depth from id (count of separators + 1) to avoid any carryover issues
    nodes <- nodes %>% mutate(depth = ifelse(id == "", 0L, stringr::str_count(id, "\\|") + 1L))
    # Carry phylum for color (second token when present)
    nodes <- nodes %>% mutate(
      # Extract phylum (2nd token) for color mapping
      phylum_for_color = purrr::map_chr(strsplit(id, "\\|", fixed = FALSE), function(parts) if (length(parts) >= 2) parts[2] else "")
    )

    # Assign angular spans iteratively per depth using cumulative fractions
    nodes <- nodes %>% arrange(depth, parent, desc(value), label)
    nodes <- nodes %>% mutate(y0 = NA_real_, y1 = NA_real_)
    # Depth 1 (root children of empty parent): spread across [0,1]
    d1 <- nodes %>% filter(depth == 1) %>% mutate(frac = value / sum(value))
    d1 <- d1 %>% mutate(y0 = lag(cumsum(frac), default = 0), y1 = cumsum(frac)) %>% select(id, y0, y1)
    # Join and coalesce into the main y0/y1
    nodes <- nodes %>%
      left_join(d1 %>% rename(y0_d1 = y0, y1_d1 = y1), by = "id") %>%
      mutate(y0 = coalesce(y0, y0_d1), y1 = coalesce(y1, y1_d1)) %>%
      select(-y0_d1, -y1_d1)
    # For deeper levels, map within each parent span
    max_d <- max(nodes$depth, na.rm = TRUE)
    if (max_d >= 2) {
      for (d in 2:max_d) {
        cur <- nodes %>% filter(depth == d)
        if (nrow(cur) == 0) next
        # parent spans
  par_spans <- nodes %>% select(parent = id, p_y0 = y0, p_y1 = y1)
  cur <- cur %>% left_join(par_spans, by = "parent")
        # compute fractions within parent and scale to parent span
        cur <- cur %>% group_by(parent) %>% mutate(frac = value / sum(value)) %>% ungroup()
        cur <- cur %>% group_by(parent) %>% mutate(cum = cumsum(frac), y0 = p_y0 + (lag(cum, default = 0)) * (p_y1 - p_y0), y1 = p_y0 + cum * (p_y1 - p_y0)) %>% ungroup()
        nodes <- nodes %>%
          left_join(cur %>% select(id, y0_new = y0, y1_new = y1), by = "id") %>%
          mutate(y0 = coalesce(y0, y0_new), y1 = coalesce(y1, y1_new)) %>%
          select(-y0_new, -y1_new)
      }
    }

    # Use citystat_core12 palette from maps.R and assign phylum colors to maximize adjacent contrast
    citystat_core12 <- c(
      Navy      = "#233B5D",
      Blue      = "#2F6DA8",
      DeepTeal  = "#1C7C7D",
      Cyan      = "#17BEBB",
      Forest    = "#2E7D32",
      Olive     = "#7A8F30",
      Mustard   = "#C7A41A",
      Orange    = "#E07A2D",
      Vermilion = "#D0432B",
      Crimson   = "#B01E2F",
      Plum      = "#7A3E9D",
      Indigo    = "#3B4BA3"
    )

    # Depth-2 phylum nodes and sizes
    phylum_nodes <- nodes %>% filter(depth == 2) %>% distinct(label, .keep_all = TRUE) %>% select(label, value) %>% arrange(desc(value))
    phyla <- phylum_nodes$label
    k <- length(phyla)
    m <- length(citystat_core12)

    # High-contrast traversal of palette indices (fixed for 12, generic fallback otherwise)
    spread_idx <- if (m >= 12) {
      c(1,7,3,9,5,11,2,8,4,10,6,12)
    } else {
      a <- 1:ceiling(m/2); b <- (ceiling(m/2)+1):m; as.integer(na.omit(as.vector(rbind(a,b))))
    }

    # Build color buckets by index following spread order, assign phyla in round-robin by descending size
    buckets <- vector("list", m); for (i in seq_len(m)) buckets[[i]] <- character(0)
    if (k > 0) {
      cur_pos <- 1
      for (ph in phyla) {
        idx <- spread_idx[(cur_pos - 1) %% length(spread_idx) + 1]
        buckets[[idx]] <- c(buckets[[idx]], ph)
        cur_pos <- cur_pos + 1
      }
    }

    # Build final phylum order by cycling through buckets; this spaces identical colors apart
    final_order <- character(0)
    remaining <- sum(lengths(buckets))
    while (remaining > 0) {
      for (idx in spread_idx) {
        if (length(buckets[[idx]]) > 0) {
          final_order <- c(final_order, buckets[[idx]][1])
          buckets[[idx]] <- buckets[[idx]][-1]
          remaining <- remaining - 1
        }
      }
    }
    phylum_order_rank <- setNames(seq_along(final_order), final_order)

    # Phylum -> color mapping, cycling palette if more phyla than colors
    pal_cycle <- rep(unname(citystat_core12[spread_idx]), length.out = k)
    phylum_colors <- setNames(pal_cycle[seq_len(k)], final_order)

    # Recompute angular spans for depth 2 using our contrast order within each parent
    if (max_d >= 2) {
      # Redo depth-2 spans explicitly to honor the custom order
      cur <- nodes %>% filter(depth == 2) %>% mutate(ord = phylum_order_rank[.data$label]) %>% arrange(.data$parent, .data$ord)
      par_spans <- nodes %>% select(parent = id, p_y0 = y0, p_y1 = y1)
      cur <- cur %>% left_join(par_spans, by = "parent")
      cur <- cur %>% group_by(parent) %>% mutate(frac = value / sum(value)) %>% ungroup()
      cur <- cur %>% group_by(parent) %>% mutate(cum = cumsum(frac), y0 = p_y0 + (lag(cum, default = 0)) * (p_y1 - p_y0), y1 = p_y0 + cum * (p_y1 - p_y0)) %>% ungroup()
      nodes <- nodes %>%
        left_join(cur %>% select(id, y0_new = y0, y1_new = y1), by = "id") %>%
        mutate(y0 = coalesce(y0, y0_new), y1 = coalesce(y1, y1_new)) %>%
        select(-y0_new, -y1_new)
    }

    # Assign fill color from phylum palette for all depths (constant per phylum)
    nodes <- nodes %>% mutate(
      fill_col = vapply(seq_len(nrow(.)), function(i) {
        ph <- phylum_for_color[i]
        if (is.na(y0[i]) || is.na(y1[i])) return(NA_character_)
        if (ph == "" || !(ph %in% names(phylum_colors))) return("#BBBBBB")
        phylum_colors[[ph]]
      }, character(1))
    )

  # Write nodes TSV for quick inspection
  write_tsv(nodes %>% select(id, parent, label, depth, value, y0, y1, phylum_for_color, fill_col),
        file.path("mag_analysis", "mag_taxonomy_sunburst_nodes.tsv"))

    # Draw static multi-ring sunburst using circular arcs if ggforce is available; fallback to polar-rects otherwise
    max_depth <- max(nodes$depth, na.rm = TRUE)
    if (requireNamespace("ggforce", quietly = TRUE)) {
      nodes_arc <- nodes %>% mutate(
        start = (y0 * 2 * pi) - pi/2,
        end   = (y1 * 2 * pi) - pi/2,
        r0    = pmax(depth - 1, 0),
        r     = depth - 1e-3
      )
      p_sun <- ggplot() +
        ggforce::geom_arc_bar(
          data = nodes_arc,
          aes(x0 = 0, y0 = 0, r0 = r0, r = r, start = start, end = end, fill = fill_col),
          color = "white", linewidth = 0.8, na.rm = TRUE
        ) +
        scale_fill_identity(guide = "none") +
        coord_fixed() +
        theme_void() +
        theme(plot.margin = grid::unit(rep(4, 4), "pt")) +
        labs(title = "High-quality MAG taxonomy (sunburst)")

      # Phylum labels placed at mid-angle on ring ~1.5 radius
      phyl_labels <- nodes_arc %>%
        filter(depth == 2) %>%
        mutate(w = y1 - y0, mid = (start + end) / 2, rx = 1.5 * cos(mid), ry = 1.5 * sin(mid)) %>%
        filter(w >= 0.02)
      if (nrow(phyl_labels) > 0) {
        p_sun <- p_sun +
          geom_text(data = phyl_labels, aes(x = rx, y = ry, label = label), inherit.aes = FALSE,
                    size = 3.2, color = "black", check_overlap = TRUE)
      }
    } else {
      p_sun <- ggplot(nodes, aes(ymin = y0, ymax = y1, xmin = depth - 1, xmax = depth, fill = fill_col)) +
        geom_rect(color = "white", linewidth = 0.8, na.rm = TRUE) +
        coord_polar(theta = "y") +
        scale_fill_identity(guide = "none") +
        scale_x_continuous(limits = c(0, max_depth), expand = c(0, 0)) +
        scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
        theme_void() +
        theme(plot.margin = grid::unit(rep(4, 4), "pt")) +
        labs(title = "High-quality MAG taxonomy (sunburst)")

      phyl_labels <- nodes %>%
        filter(depth == 2) %>%
        mutate(w = y1 - y0) %>%
        filter(w >= 0.02) %>%
        mutate(x = 1.5, y = (y0 + y1) / 2)
      if (nrow(phyl_labels) > 0) {
        p_sun <- p_sun +
          geom_text(data = phyl_labels, aes(x = x, y = y, label = label), inherit.aes = FALSE,
                    size = 3.2, color = "black", check_overlap = TRUE)
      }
    }

    # Save PNG
  out_png <- file.path("mag_analysis", "mag_taxonomy_sunburst.png")
  ggsave(out_png, p_sun, width = 10, height = 10, dpi = 300, bg = "white")
  message("Saved sunburst: ", normalizePath(out_png, winslash = "/", mustWork = FALSE))

    # Cleanup any prior interactive outputs and old genus bar if present
    if (file.exists(file.path("mag_analysis", "mag_taxonomy_sunburst.html"))) unlink(file.path("mag_analysis", "mag_taxonomy_sunburst.html"))
    if (dir.exists(file.path("mag_analysis", "mag_taxonomy_sunburst_files"))) unlink(file.path("mag_analysis", "mag_taxonomy_sunburst_files"), recursive = TRUE)
    if (file.exists(file.path("mag_analysis", "mag_hq_by_genus.png"))) unlink(file.path("mag_analysis", "mag_hq_by_genus.png"))
  }
}


# 10) Per-site treatment-highlighted sunburst (reuse same layout) -----------
{
  nodes_path <- file.path("mag_analysis", "mag_taxonomy_sunburst_nodes.tsv")
  if (file.exists(nodes_path)) {
    nodes_base <- readr::read_tsv(nodes_path, show_col_types = FALSE)

    # Helper: collapse deepest classified taxonomy to a single path string
    tax_to_path <- function(df) {
      ranks <- c("domain", "phylum", "class", "order", "family", "genus", "species")
      x <- df %>%
        transmute(
          domain = ifelse(is.na(domain) | domain == "", NA_character_, domain),
          phylum = ifelse(is.na(phylum) | phylum == "", NA_character_, phylum),
          class  = ifelse(is.na(class)  | class  == "", NA_character_, class),
          order  = ifelse(is.na(order)  | order  == "", NA_character_, order),
          family = ifelse(is.na(family) | family == "", NA_character_, family),
          genus  = ifelse(is.na(genus)  | genus  == "", NA_character_, genus),
          species= ifelse(is.na(species)| species== "", NA_character_, species)
        )
      # Drop rows with missing phylum to match the base sunburst
      x <- x %>% filter(!is.na(phylum))
      if (!nrow(x)) return(character(0))
      apply(x[, ranks], 1, function(row) paste(row[!is.na(row)], collapse = "|"))
    }

    # Helper: all prefix node ids for a set of paths
    all_prefix_ids <- function(paths) {
      if (!length(paths)) return(character(0))
      unlist(lapply(paths, function(p) {
        parts <- strsplit(p, "\\|", fixed = FALSE)[[1]]
        vapply(seq_along(parts), function(d) paste(parts[seq_len(d)], collapse = "|"), character(1))
      }), use.names = FALSE) |> unique()
    }

    # Colors: maps palette for treatment, neutral grey for non-highlight
    treat_cols <- c(untreated = "#D0432B", treated = citystat_core12[["Blue"]])
  neutral_grey <- "#8C9392"

    # Build HQ-MAG taxonomy paths per site (sample_code) and treatment
    hq <- unified %>%
      filter(classification_mimag == "High-quality MAG", !is.na(sample)) %>%
      mutate(sample_code = sub("_S[0-9]+$", "", sample))

    # Attach treatment using metadata file directly (self-contained)
    meta_path <- "PROJECT DATA MASTERLIST - PROJECT Y1 DATA.tsv"
    if (file.exists(meta_path)) {
      meta_raw <- readr::read_tsv(meta_path, show_col_types = FALSE)
      nm <- names(meta_raw) %>%
        stringr::str_replace_all("[\\r\\n]", "") %>%
        stringr::str_trim() %>%
        stringr::str_replace_all("[[:space:]]+", "_") %>%
        stringr::str_replace_all("[()'°]", "") %>%
        stringr::str_replace_all("__+", "_")
      names(meta_raw) <- nm
      if (all(c("SAMPLE_CODE", "SAMPLE_TYPE", "SAMPLE_DESCRIPTION") %in% names(meta_raw))) {
        treat_map <- meta_raw %>%
          transmute(
            sample_code = .data$SAMPLE_CODE %>% stringr::str_trim(),
            sample_type = .data$SAMPLE_TYPE,
            sample_description = .data$SAMPLE_DESCRIPTION
          ) %>%
          mutate(
            treated = dplyr::case_when(
              stringr::str_detect(sample_type, "(?i)untreated") ~ "untreated",
              stringr::str_detect(sample_type, "(?i)treated") ~ "treated",
              TRUE ~ NA_character_
            ),
            site_base = sample_description %>%
              stringr::str_replace_all("[-–—]+", " ") %>%
              stringr::str_replace_all("(?i)\\b(untreated|treated)\\b", "") %>%
              stringr::str_replace_all("\\b[0-9]+\\b", "") %>%
              stringr::str_squish()
          ) %>%
          select(sample_code, treated, site_base)
        hq <- hq %>% mutate(sample_code = stringr::str_trim(sample_code)) %>% left_join(treat_map, by = "sample_code")
        message("Section 10: distinct treated values after join: ", paste(unique(na.omit(hq$treated)), collapse = ", "))
      } else {
        hq <- hq %>% mutate(treated = NA_character_)
      }
    } else {
      hq <- hq %>% mutate(treated = NA_character_)
    }

  # Compute taxonomy paths once across all rows
  paths <- tax_to_path(hq)
  hq <- hq %>% mutate(path = paths)
    # Remove rows that produced empty path (e.g., missing phylum)
    hq <- hq %>% filter(!is.na(path), path != "")

    if (nrow(hq) > 0) {
      # Only sites (by sample name) that have both treated and untreated
      site_counts <- hq %>%
        filter(!is.na(site_base), !is.na(treated)) %>%
        distinct(site_base, treated) %>%
        count(site_base, treated) %>%
        tidyr::pivot_wider(names_from = treated, values_from = n, values_fill = 0)
      sites <- site_counts %>% filter(untreated > 0, treated > 0) %>% pull(site_base) %>% sort()
      message(sprintf("Section 10: %d HQ MAG rows, %d sites with both", nrow(hq), length(sites)))

      # Precompute arc geometry once
      max_depth <- max(nodes_base$depth, na.rm = TRUE)
      if (requireNamespace("ggforce", quietly = TRUE)) {
        nodes_arc <- nodes_base %>% mutate(
          start = (y0 * 2 * pi) - pi/2,
          end   = (y1 * 2 * pi) - pi/2,
          r0    = pmax(depth - 1, 0),
          r     = depth - 1e-3
        )
      }

      presence_rows <- list()
      presence_rows_genus <- list()
      for (site in sites) {
        site_df <- hq %>% filter(site_base == site)
        t_n <- sum(site_df$treated == "treated", na.rm = TRUE)
        u_n <- sum(site_df$treated == "untreated", na.rm = TRUE)
        message(sprintf("  Site %s: total HQ=%d (treated=%d, untreated=%d)", site, nrow(site_df), t_n, u_n))
        if (t_n == 0 || u_n == 0) {
          message("    skip site (needs both treated and untreated)")
          next
        }
        for (lvl in c("untreated", "treated")) {
          sub <- site_df %>% filter(treated == lvl)
          if (!nrow(sub)) { message(sprintf("    skip level %s (0 rows)", lvl)); next }

          # Per-site, per-treatment leaf counts and node-level relative abundance
          leaf_counts <- sub %>% count(path, name = "value")
          total <- sum(leaf_counts$value)
          if (total <= 0) { message(sprintf("    skip level %s (no counts)", lvl)); next }

          prefix_sum <- purrr::map2_dfr(leaf_counts$path, leaf_counts$value, function(p, val) {
            parts <- strsplit(p, "\\|", fixed = FALSE)[[1]]
            if (!length(parts)) return(tibble())
            ids <- vapply(seq_along(parts), function(d) paste(parts[seq_len(d)], collapse = "|"), character(1))
            tibble(id = ids, value = val)
          }) %>%
            group_by(id) %>% summarize(value = sum(value), .groups = "drop") %>%
            mutate(rel = pmax(value / total, 0))

          # Map rel abundance -> alpha [0.2, 0.9]
          prefix_sum <- prefix_sum %>% mutate(alpha_v = scales::rescale(rel, to = c(0.2, 0.9)))

          fill_df <- nodes_base %>%
            left_join(prefix_sum %>% select(id, alpha_v), by = "id") %>%
            mutate(
              fill_hi = ifelse(!is.na(alpha_v), scales::alpha(treat_cols[[lvl]], alpha_v), scales::alpha(neutral_grey, 0.2))
            )

          # Compose plot
          if (requireNamespace("ggforce", quietly = TRUE)) {
            nodes_draw <- nodes_arc %>% left_join(fill_df %>% select(id, fill_hi), by = "id")
            p <- ggplot() +
              ggforce::geom_arc_bar(
                data = nodes_draw,
                aes(x0 = 0, y0 = 0, r0 = r0, r = r, start = start, end = end, fill = fill_hi),
                color = "white", linewidth = 0.8, na.rm = TRUE
              ) +
              scale_fill_identity(guide = "none") +
              coord_fixed() +
              theme_void() +
              theme(plot.margin = grid::unit(rep(4, 4), "pt")) +
              labs(title = paste0("HQ MAG taxonomy at site ", site, " (", lvl, ")"))

            # Phylum labels at mid-angle on ring ~1.5 radius
            phyl_labels <- nodes_draw %>%
              filter(depth == 2) %>%
              mutate(w = y1 - y0, mid = (start + end) / 2, rx = 1.5 * cos(mid), ry = 1.5 * sin(mid)) %>%
              filter(w >= 0.02)
            if (nrow(phyl_labels) > 0) {
              p <- p +
                geom_text(data = phyl_labels, aes(x = rx, y = ry, label = label), inherit.aes = FALSE,
                          size = 3.0, color = "black", check_overlap = TRUE)
            }
          } else {
            nodes_draw <- nodes_base %>% left_join(fill_df %>% select(id, fill_hi), by = "id")
            p <- ggplot(nodes_draw, aes(ymin = y0, ymax = y1, xmin = depth - 1, xmax = depth, fill = fill_hi)) +
              geom_rect(color = "white", linewidth = 0.8, na.rm = TRUE) +
              coord_polar(theta = "y") +
              scale_fill_identity(guide = "none") +
              scale_x_continuous(limits = c(0, max_depth), expand = c(0, 0)) +
              scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
              theme_void() +
              theme(plot.margin = grid::unit(rep(4, 4), "pt")) +
              labs(title = paste0("HQ MAG taxonomy at site ", site, " (", lvl, ")"))

            phyl_labels <- nodes_draw %>%
              filter(depth == 2) %>% mutate(w = y1 - y0) %>% filter(w >= 0.02) %>%
              mutate(x = 1.5, y = (y0 + y1) / 2)
            if (nrow(phyl_labels) > 0) {
              p <- p +
                geom_text(data = phyl_labels, aes(x = x, y = y, label = label), inherit.aes = FALSE,
                          size = 3.0, color = "black", check_overlap = TRUE)
            }
          }

          site_safe <- gsub("[^A-Za-z0-9_.-]", "_", site)
          out_png <- file.path("mag_analysis", paste0("mag_taxonomy_sunburst_site_", site_safe, "_", lvl, ".png"))
          ggsave(out_png, p, width = 10, height = 10, dpi = 300, bg = "white")
          message("Saved site sunburst: ", normalizePath(out_png, winslash = "/", mustWork = FALSE))
        }

    # Combined presence: treated-only (Blue), untreated-only (Vermilion), both (Vermilion), none (grey)
        sub_t <- site_df %>% filter(treated == "treated")
        sub_u <- site_df %>% filter(treated == "untreated")
        t_ids <- all_prefix_ids(unique(sub_t$path))
        u_ids <- all_prefix_ids(unique(sub_u$path))
    fill_df_comb <- nodes_base %>% mutate(
          fill_cat = case_when(
            id %in% t_ids & id %in% u_ids ~ "both",
            id %in% t_ids & !(id %in% u_ids) ~ "treated",
            id %in% u_ids & !(id %in% t_ids) ~ "untreated",
            TRUE ~ "none"
          ),
          fill_hi = dplyr::case_when(
      # BOTH = Vermilion; Treated = Blue; Untreated = Vermilion; None = grey
      fill_cat == "both" ~ citystat_core12[["Vermilion"]],
            fill_cat == "treated" ~ treat_cols[["treated"]],
            fill_cat == "untreated" ~ treat_cols[["untreated"]],
            TRUE ~ neutral_grey
          )
        )

  # Debug counts per category
  cat_counts <- fill_df_comb %>% count(fill_cat, name = "n")
  message(sprintf("    combined categories: %s",
      paste(sprintf("%s=%d", cat_counts$fill_cat, cat_counts$n), collapse = ", ")))

        if (requireNamespace("ggforce", quietly = TRUE)) {
          nodes_draw <- nodes_arc %>% left_join(fill_df_comb %>% select(id, fill_hi), by = "id")
          p <- ggplot() +
            ggforce::geom_arc_bar(
              data = nodes_draw,
              aes(x0 = 0, y0 = 0, r0 = r0, r = r, start = start, end = end, fill = fill_hi),
              color = "white", linewidth = 0.8, na.rm = TRUE
            ) +
            scale_fill_identity(guide = "none") +
            coord_fixed() +
            theme_void() +
            theme(plot.margin = grid::unit(rep(4, 4), "pt")) +
            labs(title = paste0("HQ MAG taxonomy at site ", site, " (combined)"))

          phyl_labels <- nodes_draw %>%
            filter(depth == 2) %>%
            mutate(w = y1 - y0, mid = (start + end) / 2, rx = 1.5 * cos(mid), ry = 1.5 * sin(mid)) %>%
            filter(w >= 0.02)
          if (nrow(phyl_labels) > 0) {
            p <- p +
              geom_text(data = phyl_labels, aes(x = rx, y = ry, label = label), inherit.aes = FALSE,
                        size = 3.0, color = "black", check_overlap = TRUE)
          }
        } else {
          nodes_draw <- nodes_base %>% left_join(fill_df_comb %>% select(id, fill_hi), by = "id")
          p <- ggplot(nodes_draw, aes(ymin = y0, ymax = y1, xmin = depth - 1, xmax = depth, fill = fill_hi)) +
            geom_rect(color = "white", linewidth = 0.8, na.rm = TRUE) +
            coord_polar(theta = "y") +
            scale_fill_identity(guide = "none") +
            scale_x_continuous(limits = c(0, max_depth), expand = c(0, 0)) +
            scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
            theme_void() +
            theme(plot.margin = grid::unit(rep(4, 4), "pt")) +
            labs(title = paste0("HQ MAG taxonomy at site ", site, " (combined)"))

          phyl_labels <- nodes_draw %>%
            filter(depth == 2) %>% mutate(w = y1 - y0) %>% filter(w >= 0.02) %>%
            mutate(x = 1.5, y = (y0 + y1) / 2)
          if (nrow(phyl_labels) > 0) {
            p <- p +
              geom_text(data = phyl_labels, aes(x = x, y = y, label = label), inherit.aes = FALSE,
                        size = 3.0, color = "black", check_overlap = TRUE)
          }
        }

        site_safe <- gsub("[^A-Za-z0-9_.-]", "_", site)
        out_png <- file.path("mag_analysis", paste0("mag_taxonomy_sunburst_site_", site_safe, "_combined.png"))
        ggsave(out_png, p, width = 10, height = 10, dpi = 300, bg = "white")
        message("Saved site sunburst: ", normalizePath(out_png, winslash = "/", mustWork = FALSE))

        # Accumulate presence table rows for this site (node-level)
        ranks <- c("domain", "phylum", "class", "order", "family", "genus", "species")
  rank_names <- setNames(ranks, as.character(seq_along(ranks)))
        pres <- nodes_base %>%
          mutate(present_treated = id %in% t_ids, present_untreated = id %in% u_ids) %>%
          filter(present_treated | present_untreated) %>%
          transmute(
            site = site,
            id,
            label,
            depth,
            rank = dplyr::case_when(depth >= 1 & depth <= length(ranks) ~ ranks[depth], TRUE ~ NA_character_),
            present_treated,
            present_untreated,
            status = dplyr::case_when(present_treated & present_untreated ~ "both", present_treated ~ "treated_only", present_untreated ~ "untreated_only"),
            color_hex = dplyr::case_when(
              status == "both" ~ citystat_core12[["Vermilion"]],
              status == "treated_only" ~ treat_cols[["treated"]],
              status == "untreated_only" ~ treat_cols[["untreated"]],
              TRUE ~ neutral_grey
            )
          )
        presence_rows[[length(presence_rows) + 1]] <- pres

        # Accumulate genus-level presence for this site
        gens_t <- sub_t %>% filter(!is.na(genus) & genus != "") %>% distinct(genus)
        gens_u <- sub_u %>% filter(!is.na(genus) & genus != "") %>% distinct(genus)
        gens_all <- union(gens_t$genus, gens_u$genus)
        if (length(gens_all)) {
          ph_map <- site_df %>% filter(!is.na(genus), !is.na(phylum)) %>% distinct(genus, phylum)
          if (nrow(ph_map) == 0) ph_map <- unified %>% filter(!is.na(genus), !is.na(phylum)) %>% distinct(genus, phylum)
          pres_gen <- tibble(
            site = site,
            genus = gens_all,
            present_treated = gens_all %in% gens_t$genus,
            present_untreated = gens_all %in% gens_u$genus
          ) %>%
            mutate(presence_category = case_when(
              present_treated & present_untreated ~ "both",
              present_treated & !present_untreated ~ "treated_only",
              !present_treated & present_untreated ~ "untreated_only",
              TRUE ~ "none"
            )) %>%
            left_join(ph_map, by = "genus") %>%
            mutate(color_hex = case_when(
              presence_category == "both" ~ citystat_core12[["Vermilion"]],
              presence_category == "treated_only" ~ treat_cols[["treated"]],
              presence_category == "untreated_only" ~ treat_cols[["untreated"]],
              TRUE ~ neutral_grey
            ))
          presence_rows_genus[[length(presence_rows_genus) + 1]] <- pres_gen
        }
      }

      # Write combined presence table
      if (length(presence_rows)) {
        presence_tbl <- dplyr::bind_rows(presence_rows)
        readr::write_tsv(presence_tbl, file.path("mag_analysis", "mag_taxonomy_presence_by_site.tsv"))
        message("Saved presence table: ", normalizePath(file.path("mag_analysis", "mag_taxonomy_presence_by_site.tsv"), winslash = "/", mustWork = FALSE))
      }

      # Genus-level presence table (per-site)
      if (length(presence_rows_genus)) {
        presence_tbl_gen <- dplyr::bind_rows(presence_rows_genus)
        readr::write_tsv(presence_tbl_gen, file.path("mag_analysis", "mag_taxa_presence_by_site.tsv"))
        message("Saved genus presence table: ", normalizePath(file.path("mag_analysis", "mag_taxa_presence_by_site.tsv"), winslash = "/", mustWork = FALSE))
      }
    } else {
      message("Skipping site-level treatment sunbursts: no HQ MAGs found.")
    }
  } else {
    message("Skipping site-level treatment sunbursts: nodes TSV not found (run main sunburst first).")
  }
}

# 4) Genome size vs N50 colored by genus (metaWRAP N50 preferred, else CheckM2)
unified <- unified %>% mutate(N50_any = coalesce(meta_N50, Contig_N50), size_any = coalesce(meta_size, Genome_Size))
p_n50 <- ggplot(unified, aes(x = size_any, y = N50_any, color = genus)) +
  geom_point(alpha = 0.7) +
  scale_x_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  scale_color_discrete(guide = "none") +
  theme_minimal(base_size = 11) +
  labs(title = "Assembly metrics: Genome size vs N50", x = "Genome size (bp)", y = "N50 (bp)")

ggsave(file.path("mag_analysis", "mag_size_vs_n50.png"), p_n50, width=8, height=6, dpi=300, bg="white")

# 5) Optional: coverage vs contamination if RefineM coverage present
if ("refinem_mean_cov" %in% names(unified) && any(!is.na(unified$refinem_mean_cov))) {
  p_cov <- ggplot(unified, aes(x = refinem_mean_cov, y = coalesce(Contamination_chk2, Contamination_chk1), color = classification_mimag))+
    geom_point(alpha = 0.7) +
    scale_color_manual(values = classification_colors) +
    theme_minimal(base_size = 11) +
    labs(title = "RefineM mean coverage vs. contamination", x = "Mean coverage (RefineM)", y = "Contamination (%)")
  ggsave(file.path("mag_analysis", "mag_cov_vs_contam.png"), p_cov, width=8, height=6, dpi=300, bg="white")
}

 

# 6) Contigs per MAG per sample — combined violin + boxplot + scatter --------
contigs_df <- unified %>%
  mutate(contigs_any = coalesce(Total_Contigs, Contigs_chk1)) %>%
  filter(!is.na(sample), !is.na(contigs_any))

if (nrow(contigs_df) > 0) {
  # Order samples by median number of contigs (descending) for readability
  samp_order <- contigs_df %>% group_by(sample) %>% summarize(med = median(contigs_any), .groups = "drop") %>% arrange(desc(med)) %>% pull(sample)
  contigs_df <- contigs_df %>% mutate(sample = factor(sample, levels = samp_order))

  # Combined Violin + Boxplot + Scatter overlay
  p_contigs_violin_box <- ggplot(contigs_df, aes(x = sample, y = contigs_any)) +
    geom_violin(fill = "#A8BAC4", color = "grey55", alpha = 0.55, scale = "width", trim = TRUE) +
    geom_boxplot(outlier.shape = NA, width = 0.22, fill = "white", color = "grey25", alpha = 0.9, linewidth = 0.3) +
    geom_point(aes(color = classification_mimag), position = position_jitter(width = 0.16, height = 0, seed = 4), alpha = 0.55, size = 1.2) +
    scale_color_manual(values = classification_colors, name = "MIMAG") +
    scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "right") +
    labs(title = "Contigs per MAG per sample (violin + boxplot + scatter)", x = "Sample", y = "Number of contigs (log10)")

  ggsave(file.path("mag_analysis", "mag_contigs_violin_box_per_sample.png"), p_contigs_violin_box, width = 14, height = 6.5, dpi = 300, bg = "white")
}

# 7) Correlation between completeness and contig number ----------------------
corr_df <- unified %>%
  mutate(
    contigs_any = coalesce(Total_Contigs, Contigs_chk1),
    completeness_any = coalesce(Completeness_chk2, Completeness_chk1)
  ) %>%
  filter(!is.na(contigs_any), !is.na(completeness_any), contigs_any > 0)

if (nrow(corr_df) > 2) {
  overall_pearson  <- suppressWarnings(cor(corr_df$completeness_any, corr_df$contigs_any, method = "pearson"))
  overall_spearman <- suppressWarnings(cor(corr_df$completeness_any, corr_df$contigs_any, method = "spearman"))

  per_sample_corr <- corr_df %>%
    filter(!is.na(sample)) %>%
    group_by(sample) %>%
    summarize(
      n = n(),
      pearson = ifelse(n >= 5, suppressWarnings(cor(completeness_any, contigs_any, method = "pearson")), NA_real_),
      spearman = ifelse(n >= 5, suppressWarnings(cor(completeness_any, contigs_any, method = "spearman")), NA_real_),
      .groups = "drop"
    )

  # Save correlations
  corr_out <- bind_rows(
    tibble(scope = "overall", sample = NA_character_, n = nrow(corr_df), pearson = overall_pearson, spearman = overall_spearman),
    per_sample_corr %>% mutate(scope = "per-sample") %>% select(scope, sample, n, pearson, spearman)
  )
  write_tsv(corr_out, file.path("mag_analysis", "completeness_contigs_correlation.tsv"))

  # Scatter with log-scaled contigs and a linear fit (on transformed scale)
  xmax <- max(corr_df$contigs_any, na.rm = TRUE)
  ymax <- max(corr_df$completeness_any, na.rm = TRUE)
  lab  <- sprintf("Pearson r = %.2f\nSpearman rho = %.2f", overall_pearson, overall_spearman)

  p_corr <- ggplot(corr_df, aes(x = contigs_any, y = completeness_any, color = classification_mimag)) +
    geom_point(alpha = 0.5, size = 1.6) +
    scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
    scale_color_manual(values = classification_colors, name = "MIMAG") +
    geom_smooth(method = "lm", se = TRUE, color = "black") +
    annotate("text", x = xmax, y = ymax, hjust = 1, vjust = 1, label = lab, size = 3.6) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "right") +
    labs(title = "Completeness vs contig number", x = "Number of contigs (log10)", y = "Completeness (%)")

  ggsave(file.path("mag_analysis", "mag_completeness_vs_contigs.png"), p_corr, width = 8.5, height = 6, dpi = 300, bg = "white")

  # Faceted by MAG quality (classification_mimag) with per-quality annotations
  qual_stats <- corr_df %>%
    filter(!is.na(classification_mimag)) %>%
    group_by(classification_mimag) %>%
    summarize(
      n = n(),
      pearson = ifelse(n >= 5, suppressWarnings(cor(completeness_any, contigs_any, method = "pearson")), NA_real_),
      spearman = ifelse(n >= 5, suppressWarnings(cor(completeness_any, contigs_any, method = "spearman")), NA_real_),
      x_max = max(contigs_any, na.rm = TRUE),
      y_max = max(completeness_any, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(label = ifelse(is.na(pearson) | is.na(spearman),
                          paste0("n=", n),
                          sprintf("n=%d\nr=%.2f; rho=%.2f", n, pearson, spearman)))

  p_corr_f <- ggplot(corr_df, aes(x = contigs_any, y = completeness_any)) +
    geom_point(alpha = 0.5, size = 1.4, aes(color = classification_mimag)) +
    scale_color_manual(values = classification_colors, name = "MIMAG") +
    scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
    geom_smooth(method = "lm", se = TRUE, color = "black") +
    facet_wrap(~ classification_mimag, nrow = 1) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none") +
    labs(title = "Completeness vs contig number by MAG quality",
         x = "Number of contigs (log10)", y = "Completeness (%)")

  # Add per-quality labels at top-right within each facet
  p_corr_f <- p_corr_f +
    geom_text(data = qual_stats,
              aes(x = x_max, y = y_max, label = label),
              inherit.aes = FALSE, hjust = 1, vjust = 1, size = 3.3)

  ggsave(file.path("mag_analysis", "mag_completeness_vs_contigs_faceted.png"), p_corr_f, width = 12, height = 4.2, dpi = 300, bg = "white")
}

# 8) Phylogenomic completeness heatmap ---------------------------------------
# Rows: GTDB-Tk taxonomic groups (genus), Columns: metrics

heat_df <- unified %>%
  mutate(
    completeness = coalesce(Completeness_chk2, Completeness_chk1),
    contamination = coalesce(Contamination_chk2, Contamination_chk1),
    strain_heterogeneity = Strain_heterogeneity,
    markers = Markers_chk1,
    marker_sets = MarkerSets_chk1
  ) %>%
  filter(!is.na(genus)) %>%
  group_by(genus) %>%
  summarize(
    n = n(),
    completeness = mean(completeness, na.rm = TRUE),
    contamination = mean(contamination, na.rm = TRUE),
    strain_heterogeneity = mean(strain_heterogeneity, na.rm = TRUE),
    markers = mean(markers, na.rm = TRUE),
    marker_sets = mean(marker_sets, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # Keep top genera by n to avoid super tall plot
  arrange(desc(n)) %>%
  slice_head(n = 30)

if (nrow(heat_df) > 0) {
  # Prepare long format and z-score within each metric for color scale
  hm_long <- heat_df %>%
    select(-n) %>%
    pivot_longer(cols = -genus, names_to = "metric", values_to = "value") %>%
    group_by(metric) %>%
    mutate(z = if (all(is.na(value))) NA_real_ else as.numeric(scale(value))) %>%
    ungroup()

  # Metric display names
  metric_labels <- c(
    completeness = "Completeness (%)",
    contamination = "Contamination (%)",
    strain_heterogeneity = "Strain heterogeneity (%)",
    markers = "Markers (#)",
    marker_sets = "Marker sets (#)"
  )

  # Order genera by completeness descending
  gen_order <- heat_df %>% arrange(desc(completeness)) %>% pull(genus)
  hm_long <- hm_long %>% mutate(
    genus = factor(genus, levels = rev(gen_order)),
    metric = factor(metric, levels = names(metric_labels), labels = unname(metric_labels))
  )

  p_hm <- ggplot(hm_long, aes(x = metric, y = genus, fill = z)) +
    geom_tile(color = "white", linewidth = 0.2) +
    scale_fill_viridis(option = "C", na.value = "#f0f0f0", name = "z-score") +
    # Add raw value labels for readability
    geom_text(aes(label = ifelse(is.na(value), "", sprintf(ifelse(grepl("%", as.character(metric)), "%.1f", "%.0f"), value))),
              size = 2.8, color = "black") +
    theme_minimal(base_size = 10) +
    theme(
      axis.title = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1),
      panel.grid = element_blank(),
      legend.position = "right"
    ) +
    labs(title = "Phylogenomic metrics by GTDB genus (mean per genus)", subtitle = "Color = z-score within metric; label = raw mean value")

  ggsave(file.path("mag_analysis", "mag_phylogenomic_metrics_heatmap.png"), p_hm, width = 9, height = 8, dpi = 300, bg = "white")
}

# 9) Treated vs Untreated MAG diversity (keep in this script) ---------------

# Helpers
shannon_idx <- function(p) {
  p <- p[p > 0 & is.finite(p)]
  if (!length(p)) return(NA_real_)
  -sum(p * log(p))
}

base_id <- function(x) sub("_S[0-9]+$", "", x)

# Load metadata and derive treatment/env
meta_path <- "PROJECT DATA MASTERLIST - PROJECT Y1 DATA.tsv"
if (file.exists(meta_path)) {
  meta_raw <- read_tsv(meta_path, show_col_types = FALSE)
  nm <- names(meta_raw) %>%
    str_replace_all("[\\r\\n]", "") %>%
    str_trim() %>%
    str_replace_all("[[:space:]]+", "_") %>%
    str_replace_all("[()'°]", "") %>%
    str_replace_all("__+", "_")
  names(meta_raw) <- nm

  if (all(c("SAMPLE_CODE", "SAMPLE_TYPE") %in% names(meta_raw))) {
    meta <- meta_raw %>%
      transmute(
        sample_code = .data$SAMPLE_CODE,
        sample_type = .data$SAMPLE_TYPE
      ) %>%
      mutate(
        group_env = case_when(
          startsWith(sample_type, "H-")  ~ "Hospital",
          startsWith(sample_type, "CW-") ~ "Wastewater",
          startsWith(sample_type, "SW-") ~ "SurfaceWater",
          sample_type %in% c("Control", "Optimization") ~ sample_type,
          TRUE ~ NA_character_
        ),
        treated = case_when(
          str_detect(sample_type, "(?i)untreated") ~ "untreated",
          str_detect(sample_type, "(?i)treated") ~ "treated",
          TRUE ~ NA_character_
        )
      )

    # Build MAG diversity per sample from unified table
    mag_by_sample <- unified %>%
      filter(!is.na(sample)) %>%
      group_by(sample) %>%
      group_modify(~{
        df <- .x
        mag_rich <- nrow(df)
        hq_rich  <- sum(df$classification_mimag == "High-quality MAG", na.rm = TRUE)
  ph <- na.omit(df$genus)
        if (length(ph) > 0) {
          tab <- table(ph)
          p <- as.numeric(tab) / sum(tab)
          sh <- shannon_idx(p)
          sim <- 1 - sum(p^2)
          R <- length(tab)
          even <- if (R > 1) sh / log(R) else NA_real_
        } else {
          sh <- NA_real_; sim <- NA_real_; even <- NA_real_
        }
  tibble(mag_richness = mag_rich, hq_richness = hq_rich, genus_shannon = sh, genus_simpson = sim, genus_evenness = even)
      }) %>%
      ungroup() %>%
      mutate(sample_code = base_id(sample)) %>%
      left_join(meta, by = "sample_code") %>%
      filter(!is.na(group_env)) %>%
  mutate(treated = factor(treated, levels = c("untreated", "treated")))

    # Save table
    write_tsv(mag_by_sample, file.path("mag_analysis", "mag_diversity_by_sample.tsv"))

    # Per-environment Wilcoxon tests
    wilcox_metric <- function(df, col) {
      x <- df %>% filter(treated == "treated") %>% pull(all_of(col))
      y <- df %>% filter(treated == "untreated") %>% pull(all_of(col))
      if (length(x) >= 2 && length(y) >= 2) {
        pv <- suppressWarnings(tryCatch(wilcox.test(x, y, exact = FALSE)$p.value, error = function(e) NA_real_))
      } else pv <- NA_real_
      tibble(metric = col, p_value = pv, median_treated = median(x, na.rm = TRUE), median_untreated = median(y, na.rm = TRUE), effect_median_diff = median(x, na.rm = TRUE) - median(y, na.rm = TRUE))
    }

    tests <- mag_by_sample %>%
      filter(!is.na(treated)) %>%
      group_split(group_env) %>%
      map_dfr(function(df) {
        env <- unique(df$group_env)
        bind_rows(
          wilcox_metric(df, "mag_richness"),
          wilcox_metric(df, "hq_richness"),
          wilcox_metric(df, "genus_shannon"),
          wilcox_metric(df, "genus_simpson"),
          wilcox_metric(df, "genus_evenness")
        ) %>% mutate(group_env = env, .before = 1)
      })
    write_tsv(tests, file.path("mag_analysis", "mag_diversity_tests.tsv"))

    # Plotting helper
    plot_treat <- function(df, metric, ylab, fname) {
  d <- df %>% filter(!is.na(.data[[metric]]))
      if (nrow(d) == 0) return(invisible(NULL))
  cols <- c("untreated" = "#D0432B", "treated" = citystat_core12[["Blue"]])
      # p-value labels
      ymax <- d %>% group_by(group_env) %>% summarize(y = max(.data[[metric]], na.rm = TRUE), .groups = "drop")
      pvals <- tests %>% filter(metric == fname) %>% select(group_env, p_value)
      lab_df <- left_join(ymax, pvals, by = "group_env") %>% mutate(label = ifelse(is.na(p_value), "", paste0("p=", signif(p_value, 3))), y = y * 1.05)

      g <- ggplot(d, aes(x = treated, y = .data[[metric]], fill = treated)) +
        geom_violin(width = 0.9, alpha = 0.6, color = NA, trim = FALSE) +
        geom_boxplot(width = 0.22, outlier.shape = NA, alpha = 0.95) +
        geom_jitter(aes(color = treated), width = 0.08, alpha = 0.85, size = 1.6, stroke = 0.2) +
  # Keep all samples; show missing treated as its own panel level if any
  facet_wrap(~ group_env, scales = "free_y") +
        scale_fill_manual(values = cols, guide = "none") +
        scale_color_manual(values = cols, guide = "none") +
        theme_minimal(base_size = 11) +
        theme(strip.text = element_text(face = "bold"), panel.grid.minor = element_blank()) +
        labs(x = NULL, y = ylab, title = paste0(ylab, " by treatment"))
      if (nrow(lab_df)) g <- g + geom_text(data = lab_df, aes(x = 1.5, y = y, label = label), inherit.aes = FALSE, size = 3.2)
      ggsave(file.path("mag_analysis", paste0(fname, "_treated_vs_untreated.png")), g, width = 9, height = 5.5, dpi = 300, bg = "white")
    }

  # Single per-metric plots removed; using the faceted multi-index plot below

    # Faceted multi-index plot with raw scales (free_y), treatment colors
    long_df_raw <- mag_by_sample %>%
      select(sample, group_env, treated, mag_richness, hq_richness, genus_shannon, genus_simpson, genus_evenness) %>%
      pivot_longer(cols = -c(sample, group_env, treated), names_to = "metric", values_to = "value") %>%
      mutate(metric_label = recode(metric,
        mag_richness = "MAG richness",
        hq_richness = "HQ MAG richness",
        genus_shannon = "Genus Shannon",
        genus_simpson = "Genus Simpson",
        genus_evenness = "Genus evenness"
      ))

  cols_treat <- c("untreated" = "#D0432B", "treated" = citystat_core12[["Blue"]])

    # p-values per facet from tests (map metric -> metric_label)
  tests_lab <- tests %>%
      mutate(metric_label = recode(metric,
        mag_richness = "MAG richness",
        hq_richness = "HQ MAG richness",
    genus_shannon = "Genus Shannon",
    genus_simpson = "Genus Simpson",
    genus_evenness = "Genus evenness"
      )) %>%
      select(group_env, metric_label, p_value)

    ymax_facets <- long_df_raw %>%
      filter(!is.na(treated)) %>%
      group_by(group_env, metric_label) %>%
      summarize(y = max(value, na.rm = TRUE), .groups = "drop")

    lab_df2 <- left_join(ymax_facets, tests_lab, by = c("group_env", "metric_label")) %>%
      mutate(label = ifelse(is.na(p_value), "", paste0("p=", signif(p_value, 3))), y = y * 1.05)

  g_multi_raw <- ggplot(long_df_raw %>% filter(!is.na(value), !is.na(treated)), aes(x = treated, y = value, fill = treated)) +
      geom_violin(width = 0.9, alpha = 0.6, color = NA, trim = FALSE) +
      geom_boxplot(width = 0.22, outlier.shape = NA, alpha = 0.95) +
      geom_jitter(aes(color = treated), width = 0.08, alpha = 0.8, size = 1.6, stroke = 0.2) +
      facet_grid(metric_label ~ group_env, scales = "free_y") +
      scale_x_discrete(limits = c("untreated", "treated"), drop = FALSE) +
      scale_fill_manual(values = cols_treat, guide = "none") +
      scale_color_manual(values = cols_treat, guide = "none") +
      theme_minimal(base_size = 11) +
      theme(strip.text = element_text(face = "bold"), panel.grid.minor = element_blank()) +
      labs(x = NULL, y = NULL, title = "MAG diversity indices by treatment (separate axes)")

    if (nrow(lab_df2)) {
      g_multi_raw <- g_multi_raw +
        geom_text(data = lab_df2, aes(x = 1.5, y = y, label = label), inherit.aes = FALSE, size = 3.0)
    }

    ggsave(file.path("mag_analysis", "mag_diversity_multiindex_facet.png"), g_multi_raw, width = 11, height = 8, dpi = 300, bg = "white")
  }
}

