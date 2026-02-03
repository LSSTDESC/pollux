import numpy as np
import matplotlib.pyplot as plt
import sys

def compute_averages(data_dict, sim_list, nshells_list, N_part, extra_info, 
                     seed_list=None, catalog_type_list=None):

    if catalog_type_list is None:
        catalog_type_list = [None]  # Handle missing catalog types

    # Step 1: Compute averages and standard deviations over seeds (if applicable)
    if seed_list is not None:
        for catalog_type in catalog_type_list:
            for i_sim in sim_list:
                for i_nshells in nshells_list:
                    key = tuple(filter(lambda x: x is not None, (N_part, i_sim, seed_list[0], i_nshells, extra_info, catalog_type)))
                    tracer_pairs = data_dict[key].keys()
                    
                    avg_data = {}
                    std_data = {}

                    for tracer_pair in tracer_pairs:
                        data_list = [
                            data_dict[tuple(filter(lambda x: x is not None, (N_part, i_sim, i_seed, i_nshells, extra_info, catalog_type)))][tracer_pair]
                            for i_seed in seed_list
                        ]

                        avg_data[tracer_pair] = np.mean(data_list, axis=0)
                        std_data[tracer_pair] = np.std(data_list, axis=0, ddof=0) / np.sqrt(len(data_list))

                    avg_key = tuple(filter(lambda x: x is not None, (N_part, i_sim, "avg", i_nshells, extra_info, catalog_type)))
                    std_key = tuple(filter(lambda x: x is not None, (N_part, i_sim, "std", i_nshells, extra_info, catalog_type)))

                    data_dict[avg_key] = {tracer_pair: avg_data[tracer_pair] for tracer_pair in tracer_pairs}
                    data_dict[std_key] = {tracer_pair: std_data[tracer_pair] for tracer_pair in tracer_pairs}

                    print(f"Stored avg over seeds for {avg_key}")
                    print(f"Stored std over seeds for {std_key}")

    print()

    # Step 2: Compute averages and standard deviations over both seeds and simulations
    if seed_list is not None:
        for catalog_type in catalog_type_list:
            for i_nshells in nshells_list:
                key = tuple(filter(lambda x: x is not None, (N_part, sim_list[0], seed_list[0], i_nshells, extra_info, catalog_type)))
                tracer_pairs = data_dict[key].keys()
                
                avg_data = {}
                std_data = {}

                for tracer_pair in tracer_pairs:
                    data_list = [
                        data_dict[tuple(filter(lambda x: x is not None, (N_part, i_sim, i_seed, i_nshells, extra_info, catalog_type)))][tracer_pair]
                        for i_seed in seed_list for i_sim in sim_list
                    ]
                    
                    avg_data[tracer_pair] = np.mean(data_list, axis=0)
                    std_data[tracer_pair] = np.std(data_list, axis=0, ddof=0) / np.sqrt(len(data_list))

                avg_key = tuple(filter(lambda x: x is not None, (N_part, "avg", "avg", i_nshells, extra_info, catalog_type)))
                std_key = tuple(filter(lambda x: x is not None, (N_part, "std", "std", i_nshells, extra_info, catalog_type)))

                data_dict[avg_key] = {tracer_pair: avg_data[tracer_pair] for tracer_pair in tracer_pairs}
                data_dict[std_key] = {tracer_pair: std_data[tracer_pair] for tracer_pair in tracer_pairs}

                print(f"Stored avg over seeds for {avg_key}")
                print(f"Stored std over seeds for {std_key}")
    else:
        for catalog_type in catalog_type_list:
            for i_nshells in nshells_list:
                key = tuple(filter(lambda x: x is not None, (N_part, sim_list[0], i_nshells, extra_info, catalog_type)))
                tracer_pairs = data_dict[key].keys()
                
                avg_data = {}
                std_data = {}

                for tracer_pair in tracer_pairs:
                    data_list = [
                        data_dict[tuple(filter(lambda x: x is not None, (N_part, i_sim, i_nshells, extra_info, catalog_type)))][tracer_pair]
                        for i_sim in sim_list
                    ]
                    
                    avg_data[tracer_pair] = np.mean(data_list, axis=0)
                    std_data[tracer_pair] = np.std(data_list, axis=0, ddof=0) / np.sqrt(len(data_list))

                avg_key = tuple(filter(lambda x: x is not None, (N_part, "avg", i_nshells, extra_info, catalog_type)))
                std_key = tuple(filter(lambda x: x is not None, (N_part, "std", i_nshells, extra_info, catalog_type)))

                data_dict[avg_key] = {tracer_pair: avg_data[tracer_pair] for tracer_pair in tracer_pairs}
                data_dict[std_key] = {tracer_pair: std_data[tracer_pair] for tracer_pair in tracer_pairs}

                print(f"Stored avg over seeds for {avg_key}")
                print(f"Stored std over seeds for {std_key}")

    print()

###########################
###########################
# Plots for tomographic Cls
###########################
###########################

def replace_avg_with_std(label):
    if isinstance(label, (tuple, list)):
        if not any(isinstance(x, str) and 'avg' in x for x in label):
            raise ValueError(f"'avg' not found in label: {label}")
        return tuple(x.replace('avg', 'std') if isinstance(x, str) else x for x in label)
    elif isinstance(label, str):
        if 'avg' not in label:
            raise ValueError(f"'avg' not found in label: {label}")
        return label.replace('avg', 'std')
    else:
        raise TypeError(f"Unsupported label type: {type(label)}")

def plot_Cls_kappa(option, map_list, Cls_dict, Cls_th, ell_min_dict, list_plot, Cls_th_nosigmae=None, shotnoise=None, filename_extra=None, use_keys_for_labels=False, ylabel=None, ylim=None):

    ell_init = 0
    
    nbins = len(map_list)
    fig, axes = plt.subplots(nbins, nbins, figsize=(4 * nbins, 3 * nbins), sharey="row")

    if not use_keys_for_labels:

        title_parts = []
    
        N_part_added = False
        N_shells_added = False
        method_added = False
        
        if all(t[0] == list_plot[0][0] for t in list_plot):
            title_parts.append(fr"N_{{\rm part}}={list_plot[0][0]}^3")
            N_part_added = True
        if all(t[3] == list_plot[0][3] for t in list_plot):
            title_parts.append(fr"N_{{\rm shells}}={list_plot[0][3]}")
            N_shells_added = True
        if all(t[4] == list_plot[0][4] for t in list_plot):
            title_parts.append(fr"{{\rm {list_plot[0][4]}}}")
            method_added = True
        
        title = "$" + r",\ ".join(title_parts) + "$"
    
        fig.suptitle(title, fontsize=30)
    
        nshells_max = max(t[3] for t in list_plot)
    
    num_labels = len(list_plot)
    if option in [4, 5, 7, 9, 10]:
        if option in [4, 7]:
            num_labels -= 1
        elif option in [5, 9, 10]:
            num_labels /= 2
            num_labels = int(num_labels)
        if option in [5, 9, 10]:
            colormap = plt.cm.cool
            if num_labels == 1:
                colormap = plt.cm.cool_r
        elif option in [7]:
            colormap = plt.cm.spring
        else:
            colormap = plt.cm.winter_r
    else:
        colormap = plt.cm.cool
        # colormap = plt.cm.inferno
        # colormap = plt.cm.brg
    
    norm = plt.Normalize(0, num_labels - 1)
    values = np.linspace(norm.vmin, norm.vmax, num_labels)
    normalized_values = norm(values)
    # colors = colormap(normalized_values)

    scaled_values = 0.1 + normalized_values * 0.8
    colors = colormap(scaled_values)

    if num_labels == 1:
        colors = ["red"]

    for i, map1 in enumerate(map_list):
        for j, map2 in enumerate(map_list):
            if map1 >= map2:
                ax = axes[i, j]
                keyword = f"({map1}, {map2})"
                keyword_title = f"({map1+1}, {map2+1})"
                
                if option == 1:
                    # Plot C_l and theory comparison
                    for idx, label in enumerate(list_plot):
                        Cls = Cls_dict[label]
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                            if not method_added:
                                label_parts.append(fr"{{\rm {label[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                        color = colors[idx]
                        ax.errorbar(Cls[keyword][ell_init:, 0], Cls[keyword][ell_init:, 1], 
                                    Cls_th[2][(map1, map2)][ell_init:], fmt=".", capsize=3, label=label_plot, color=color)
                    ax.loglog(Cls_th[0][ell_init:], Cls_th[1][(map1, map2)][ell_init:], "-", label="Theory", color="black")
                    if shotnoise is not None:
                        ax.axhline(shotnoise[keyword], color="k", ls="--")
                
                elif option == 2:
                    # Plot relative error (C_l / C_l^th - 1)
                    for idx, label in enumerate(list_plot):
                        Cls = Cls_dict[label]
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                            if not method_added:
                                label_parts.append(fr"{{\rm {label[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                        color = colors[idx]
                        ax.errorbar(Cls[keyword][ell_init:, 0], 
                                    Cls[keyword][ell_init:, 1] / Cls_th[1][(map1, map2)][ell_init:] - 1,
                                    Cls_th[2][(map1, map2)][ell_init:] / Cls_th[1][(map1, map2)][ell_init:], 
                                    fmt=".", capsize=3, label=label_plot, color=color)
                    ax.axhspan(-0.05, 0.05, color="lightgreen", alpha=0.2)
                    ax.axhline(0, color="k", ls="--")
                    ax.set_ylim([-0.5, 0.5])
                    if shotnoise is not None:
                        ax.plot(Cls[keyword][ell_init:, 0], shotnoise[keyword] / Cls_th[1][(map1, map2)][ell_init:] - 1, color="k", ls="--")
                
                elif option == 3:
                    # Plot normalized difference ((C_l - C_l^th) / sigma)
                    for idx, label in enumerate(list_plot):
                        Cls = Cls_dict[label]
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                            if not method_added:
                                label_parts.append(fr"{{\rm {label[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                        color = colors[idx]
                        try:
                            std = Cls_dict[replace_avg_with_std(label)]
                            ax.errorbar(Cls[keyword][ell_init:, 0], 
                                        (Cls[keyword][ell_init:, 1] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[2][(map1, map2)][ell_init:],
                                        std[keyword][ell_init:, 1] / Cls_th[2][(map1, map2)][ell_init:],
                                        fmt=".", capsize=3, label=label_plot)#, color=color)
                        except:
                            ax.plot(Cls[keyword][ell_init:, 0], 
                                    (Cls[keyword][ell_init:, 1] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[2][(map1, map2)][ell_init:],
                                    ".-", label=label_plot)#, color=color)

                    if Cls_th_nosigmae is not None:
                        ax.fill_between(
                            Cls[keyword][ell_init:, 0],
                            -Cls_th_nosigmae[2][(map1, map2)][ell_init:] / Cls_th[2][(map1, map2)][ell_init:],
                             Cls_th_nosigmae[2][(map1, map2)][ell_init:] / Cls_th[2][(map1, map2)][ell_init:],
                            color="cyan",
                            alpha=0.3
                        )

                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)
                    # ax.set_ylim([-1.5, 1.5])
                    if shotnoise is not None:
                        ymin, ymax = ax.get_ylim()
                        ax.plot(
                            Cls[keyword][ell_init:, 0],
                            (shotnoise[keyword] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[2][(map1, map2)][ell_init:],
                            color="k", ls="--")
                        ax.set_ylim(ymin, ymax)
                
                elif option == 4:
                    # Plot delta between labels (Delta C_l / sigma). Everything is compared to nshells_max
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_b[3] == nshells_max:
                                Cls_a = Cls_dict[label_a]
                                Cls_b = Cls_dict[label_b]
                                # std_a = Cls_dict[replace_avg_with_std(label_a)]
                                # std_b = Cls_dict[replace_avg_with_std(label_b)]

                                label_parts = []
                                if not N_part_added:
                                    label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                                if not N_shells_added:
                                    label_parts.append(fr"N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]}")
                                else:
                                    print("There is something weird going on, revise the code!")
                                    sys.exit()
                                if not method_added:
                                    label_parts.append(fr"{{\rm {label_a[4]}}}")
                                label_plot = "$" + r",\ ".join(label_parts) + "$"
                                
                                color = colors[a]
                                # ax.errorbar(Cls_a[keyword][ell_init:, 0], 
                                #             ((Cls_a[keyword][ell_init:, 1] - Cls_b[keyword][ell_init:, 1]) / Cls_th[2][(map1, map2)])[ell_init:], 
                                #             np.sqrt(std_a[keyword][ell_init:, 1]**2 + std_b[keyword][ell_init:, 1]**2) / Cls_th[2][(map1, map2)][ell_init:],
                                #             fmt=".", capsize=3, label=label_plot, color=color)
                                ax.plot(Cls_a[keyword][ell_init:, 0], 
                                        ((Cls_a[keyword][ell_init:, 1] - Cls_b[keyword][ell_init:, 1]) / Cls_th[2][(map1, map2)])[ell_init:], 
                                        ".-", label=label_plot, color=color)
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)
                    if shotnoise is not None:
                        ymin, ymax = ax.get_ylim()
                        ax.plot(
                            Cls_a[keyword][ell_init:, 0],
                            (shotnoise[keyword] - Cls_b[keyword][ell_init:, 1]) / Cls_th[2][(map1, map2)][ell_init:],
                            color="k", ls="--")
                        ax.set_ylim(ymin, ymax)

                elif option == 5:
                    # Plot delta between labels (Delta C_l / sigma). Comparison between models
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_a[3] == label_b[3]:
                                Cls_a = Cls_dict[label_a]
                                Cls_b = Cls_dict[label_b]
                                # std_a = Cls_dict[replace_avg_with_std(label_a)]
                                # std_b = Cls_dict[replace_avg_with_std(label_b)]

                                label_parts = []
                                if not N_part_added:
                                    label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                                if not N_shells_added:
                                    label_parts.append(fr"N_{{\rm shells}}={label_a[3]}")
                                if not method_added:
                                    title_parts.append(fr"{{\rm {label_a[4]}}}\ {{\rm vs.}}\ {{\rm {label_b[4]}}}")
                                    title = "$" + r",\ ".join(title_parts) + "$"
                                    fig.suptitle(title, fontsize=30)
                                    method_added = True
                                label_plot = "$" + r",\ ".join(label_parts) + "$"
                                
                                color = colors[a]
                                # ax.errorbar(Cls_a[keyword][ell_init:, 0], 
                                #             ((Cls_a[keyword][ell_init:, 1] - Cls_b[keyword][ell_init:, 1]) / Cls_th[2][(map1, map2)])[ell_init:], 
                                #             np.sqrt(std_a[keyword][ell_init:, 1]**2 + std_b[keyword][ell_init:, 1]**2) / Cls_th[2][(map1, map2)][ell_init:],
                                #             fmt=".", capsize=3, label=label_plot, color=color)
                                ax.plot(Cls_a[keyword][ell_init:, 0], 
                                        ((Cls_a[keyword][ell_init:, 1] - Cls_b[keyword][ell_init:, 1]) / Cls_th[2][(map1, map2)])[ell_init:], 
                                        ".-", label=label_plot, color=color)
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)

                elif option == 6:
                    # Plot C_l and theory comparison for the downsampling plot
                    for idx, label in enumerate(list_plot):
                        Cls = Cls_dict[label]
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_plot = fr"$N_{{\rm part}}={label[0]}^3,\ N_{{\rm shells}}={label[3]}$, {label[4]}"
                        color = colors[idx]
                        ax.errorbar(Cls[keyword][ell_init:, 0], Cls[keyword][ell_init:, 1], 
                                    Cls_th[2][(map1, map2)][ell_init:], fmt=".", capsize=3, label=label_plot, color=color)
                    ax.loglog(Cls_th[0][ell_init:], Cls_th[1][(map1, map2)][ell_init:], "-", label="Theory", color="black")

                elif option == 7:
                    # Plot delta between labels (Delta C_l / sigma). Comparison between downsamplings
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and a == 0 and label_a[3] == label_b[3]:
                                Cls_a = Cls_dict[label_a]
                                Cls_b = Cls_dict[label_b]
                                # std_a = Cls_dict[replace_avg_with_std(label_a)]
                                # std_b = Cls_dict[replace_avg_with_std(label_b)]

                                label_parts = []
                                if not N_part_added:
                                    label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                                if not N_shells_added:
                                    label_parts.append(fr"N_{{\rm shells}}={label_a[3]}")
                                if not method_added:
                                    label_parts.append(fr"{{\rm {label_a[4]}}}\ {{\rm vs.}}\ {{\rm {label_b[4]}}}")
                                else:
                                    print("There is something weird going on, revise the code!")
                                    sys.exit()
                                label_plot = "$" + r",\ ".join(label_parts) + "$"
                                
                                color = colors[b - 1]
                                # ax.errorbar(Cls_a[keyword][ell_init:, 0], 
                                #             ((Cls_a[keyword][ell_init:, 1] - Cls_b[keyword][ell_init:, 1]) / Cls_th[2][(map1, map2)])[ell_init:], 
                                #             np.sqrt(std_a[keyword][ell_init:, 1]**2 + std_b[keyword][ell_init:, 1]**2) / Cls_th[2][(map1, map2)][ell_init:],
                                #             fmt=".", capsize=3, label=label_plot, color=color)
                                ax.plot(Cls_a[keyword][ell_init:, 0], 
                                        ((Cls_a[keyword][ell_init:, 1] - Cls_b[keyword][ell_init:, 1]) / Cls_th[2][(map1, map2)])[ell_init:], 
                                        ".-", label=label_plot, color=color)
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)

                elif option == 8:
                    # Plot ratio between labels. Comparison between models
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_a[3] == label_b[3]:
                                Cls_a = Cls_dict[label_a]
                                Cls_b = Cls_dict[label_b]
                                label_plot = fr"$N_{{\rm shells}}={label_a[3]},\ {{\rm {label_a[4]}}}\ {{\rm vs.}}\ {{\rm {label_b[4]}}}$"
                                color = colors[a]
                                ax.plot(Cls_a[keyword][ell_init:, 0], 
                                        (Cls_b[keyword][ell_init:, 1] / Cls_a[keyword][ell_init:, 1]), 
                                        ".-", label=label_plot, color=color)
                    ax.axhline(1, color="k", ls="--")

                elif option == 9:
                    # Plot normalized difference ((C_l - C_l^th) / sigma)
                    for idx, label in enumerate(list_plot):
                        Cls = Cls_dict[label]
                        # std = Cls_dict[replace_avg_with_std(label)]
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                            if not method_added:
                                label_parts.append(fr"{{\rm {label[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                        color = colors[idx // 2]
                        ls = ".-"
                        if label[4] == "method\ (2)":
                            ls = "--"
                        ax.plot(Cls[keyword][ell_init:, 0], 
                                (Cls[keyword][ell_init:, 1] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[2][(map1, map2)][ell_init:],
                                ls, label=label_plot, color=color)
                        ymin, ymax = ax.get_ylim()
                        if ymax > 3:
                            ax.set_ylim(ymin, 3)
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)

                elif option == 10:
                    # Plot normalized difference ((C_l - C_l^th) / sigma)
                    for idx, label in enumerate(list_plot):
                        Cls = Cls_dict[label]
                        std = Cls_dict[replace_avg_with_std(label)]
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                            if not method_added:
                                label_parts.append(fr"{{\rm {label[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                        color = colors[idx // 2]
                        shift_ell = 1
                        if label[0] == 1024:
                            ls = "--"
                            color = "blue"
                            shift_ell = 1.1
                        try:
                            std = Cls_dict[replace_avg_with_std(label)]
                            ax.errorbar(shift_ell * Cls[keyword][ell_init:, 0], 
                                        (Cls[keyword][ell_init:, 1] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[2][(map1, map2)][ell_init:],
                                        std[keyword][ell_init:, 1] / Cls_th[2][(map1, map2)][ell_init:],
                                        fmt=".", capsize=3, label=label_plot, color=color)
                        except:
                            ax.plot(Cls[keyword][ell_init:, 0], 
                                    (Cls[keyword][ell_init:, 1] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[2][(map1, map2)][ell_init:],
                                    ls, label=label_plot, color=color)
                        ymin, ymax = ax.get_ylim()
                        if ymax > 3:
                            ax.set_ylim(ymin, 3)
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)

                # Common settings for all options
                ax.set_xscale("log")
                x_limits = ax.get_xlim()
                ax.axvspan(-np.inf, max(ell_min_dict[map1], ell_min_dict[map2]), color="grey", alpha=0.3)
                ax.set_xlim(x_limits)
                if ylim is not None:
                    ax.set_ylim(ylim[0], ylim[1])
                if i == nbins - 1:
                    ax.set_xlabel(r"$\ell$", fontsize=22)
                else:
                    ax.set_xticks([])  
                if j == 0:
                    if ylabel is None:
                        ylabel = {
                            1: r"$C_\ell$",
                            2: r"$C_\ell/C_\ell^{\rm th}-1$",
                            3: r"$(C_\ell-C_\ell^{\rm th})/\sigma_\ell$",
                            4: r"$\Delta C_\ell/\sigma_\ell$",
                            5: r"$\Delta C_\ell/\sigma_\ell$",
                            6: r"$C_\ell$",
                            7: r"$\Delta C_\ell/\sigma_\ell$",
                            8: r"$C_\ell^{\rm baryon}/C_\ell^{\rm original}$",
                            9: r"$(C_\ell-C_\ell^{\rm th})/\sigma_\ell$",
                            10: r"$(C_\ell-C_\ell^{\rm th})/\sigma_\ell$",
                        }[option]
                        ax.set_ylabel(ylabel, fontsize=22)
                    else:
                        ax.set_ylabel(ylabel, fontsize=22)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_title(keyword_title, fontsize=20)
            else:
                axes[i, j].axis("off")
    
    plt.tight_layout()
    plt.subplot(nbins, nbins, 1)
    plt.legend(bbox_to_anchor=(nbins + 0.3, 1), loc="upper right", fontsize=22)
    if filename_extra is not None:
        plt.savefig(f"plots/Cell_all_{option}_{filename_extra}.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"plots/Cell_all_{option}.pdf", bbox_inches="tight")

def plot_Cls_kappa_th(option, map_list, Cls_th_dict, list_plot, ell_max_dict):
    
    nbins = len(map_list)
    ell_init = 0
    fig, axes = plt.subplots(nbins, nbins, figsize=(4 * nbins, 3 * nbins), sharey="row")
    
    num_labels = len(list_plot)
    if option in [2]:
        num_labels -= 1
        colormap = plt.cm.winter_r
    else:
        colormap = plt.cm.cool
        # colormap = plt.cm.inferno
        # colormap = plt.cm.brg

    norm = plt.Normalize(0, num_labels - 1)
    values = np.linspace(norm.vmin, norm.vmax, num_labels)
    normalized_values = norm(values)
    # colors = colormap(normalized_values)

    scaled_values = 0.1 + normalized_values * 0.8
    colors = colormap(scaled_values)
    
    for i, map1 in enumerate(map_list):
        for j, map2 in enumerate(map_list):
            if map1 >= map2:
                ax = axes[i, j]
                keyword = f"({map1}, {map2})"
                keyword_title = f"({map1+1}, {map2+1})"
                
                if option == 1:
                    # Plot C_l and theory comparison
                    for idx, label in enumerate(list_plot):
                        Cls_th = Cls_th_dict[label]
                        label_plot = label
                        color = colors[idx]
                        ax.semilogy(Cls_th[keyword][ell_init:, 0], Cls_th[keyword][ell_init:, 1], label=label_plot, color=color)

                elif option == 2:
                    # Plot delta between labels (Delta C_l / sigma). Everything is compared to halofit
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_b == "halofit":
                                Cls_a = Cls_th_dict[label_a]
                                Cls_b = Cls_th_dict[label_b]
                                label_plot = f"{label_a}/{label_b}"
                                color = colors[a]
                                ax.plot(Cls_a[keyword][ell_init:, 0], 
                                        (Cls_a[keyword][ell_init:, 1] - Cls_b[keyword][ell_init:, 1]) / Cls_b[keyword][ell_init:, 1], 
                                        ".-", label=label_plot, color=color)
                    ax.axhline(0, color="k", ls="--")
                    ax.set_ylim([-0.05, 0.05])

                # Common settings for all options
                ax.set_xscale("log")
                x_limits = ax.get_xlim()
                ax.axvspan(min(ell_max_dict[map1], ell_max_dict[map2]), x_limits[1], color="grey", alpha=0.3)
                ax.set_xlim(x_limits)
                if i == nbins - 1:
                    ax.set_xlabel(r"$\ell$", fontsize=22)
                else:
                    ax.set_xticks([])  
                if j == 0:
                    ylabel = {
                        1: r"$C_\ell^{\rm th}$",
                        2: r"$C_\ell^{\rm th}/C_\ell^{\rm halofit}-1$",
                    }[option]
                    ax.set_ylabel(ylabel, fontsize=22)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_title(keyword_title, fontsize=20)
            else:
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.subplot(nbins, nbins, 1)
    plt.legend(bbox_to_anchor=(nbins + 0.3, 1), loc="upper right", fontsize=22)

    plt.savefig(f"plots/Cell_th_{option}.pdf", bbox_inches="tight")

def plot_Cls_zsnapshots(option, map_list, z_list, z_edges_list, Cls_dict, Cls_th, ell_min_dict, ell_max_dict, list_plot, kappa=False, filename_extra=None, use_keys_for_labels=False, ylabel=None):
    
    type_ = "kappa" if kappa else "density"
    ell_init = 0

    ncols = 1
    nrows = -(-len(map_list) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey="row")
    axes = np.atleast_1d(axes).ravel()

    if not use_keys_for_labels:
    
        title_parts = []
    
        N_part_added = False
        N_shells_added = False
        method_added = False
        
        if all(t[0] == list_plot[0][0] for t in list_plot):
            title_parts.append(fr"N_{{\rm part}}={list_plot[0][0]}^3")
            N_part_added = True
        if all(t[3] == list_plot[0][3] for t in list_plot):
            title_parts.append(fr"N_{{\rm shells}}={list_plot[0][3]}")
            N_shells_added = True
        if all(t[4] == list_plot[0][4] for t in list_plot):
            title_parts.append(fr"{{\rm {list_plot[0][4]}}}")
            method_added = True
        
        title = "$" + r",\ ".join(title_parts) + "$"
    
        fig.suptitle(title, fontsize=30)

    for i, (map1, map2) in enumerate(zip(map_list, map_list)):
        if map1 == map2:
            row, col = divmod(i, ncols)
            ax = axes[i]
            keyword = f"({map1}, {map2})"
            
            # try:
            if option == 1:
                # Plot C_l and theory comparison
                for label in list_plot:
                    Cls = Cls_dict[label]
                    if use_keys_for_labels:
                        label_plot = label
                    else:
                        label_parts = []
                        if not N_part_added:
                            label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                        if not N_shells_added:
                            label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                        if not method_added:
                            label_parts.append(fr"{{\rm {label[4]}}}")
                        label_plot = "$" + r",\ ".join(label_parts) + "$"
                    if label_plot == "$$":
                        label_plot = ""
                    ax.errorbar(
                        Cls[keyword][ell_init:, 0], 
                        Cls[keyword][ell_init:, 1], 
                        Cls_th[2][(map1, map2)][ell_init:], 
                        fmt=".", 
                        capsize=3, 
                        label=label_plot
                    )
                ax.loglog(
                    Cls_th[0][ell_init:], 
                    Cls_th[1][(map1, map2)][ell_init:], 
                    "-", 
                    label="Theory",
                    color="black"
                )
                x_limits = ax.get_xlim()
                ax.axvspan(-np.inf, max(ell_min_dict[map1], ell_min_dict[map2]), color="grey", alpha=0.3)
                if ell_max_dict is not None:
                    ax.axvspan(min(ell_max_dict[map1], ell_max_dict[map2]), x_limits[1], color="grey", alpha=0.3)
                ax.set_xlim(x_limits)
                # Determine if this is the last available row in its column
                next_in_col = i + ncols
                if next_in_col >= len(map_list):  # No more plots below in this column
                    ax.set_xlabel(r"$\ell$", fontsize=22)
                else:
                    ax.set_xticklabels([])

                if col == 0:
                    if ylabel is None:
                        ax.set_ylabel(r"$C_\ell$", fontsize=22)
                    else:
                        ax.set_ylabel(ylabel, fontsize=22)
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
                ax.set_title(fr"$z_{{\rm sn}}={z_list[i]:.2f},\, z\in[{z_edges_list[i][0]:.2f},{z_edges_list[i][1]:.2f}]$", fontsize=20)
                
            elif option == 2:
                # Plot normalized difference ((C_l - C_l^th) / C_l^th)
                for label in list_plot:
                    Cls = Cls_dict[label]
                    if use_keys_for_labels:
                        label_plot = label
                    else:
                        label_parts = []
                        if not N_part_added:
                            label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                        if not N_shells_added:
                            label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                        if not method_added:
                            label_parts.append(fr"{{\rm {label[4]}}}")
                        label_plot = "$" + r",\ ".join(label_parts) + "$"
                    if label_plot == "$$":
                        label_plot = ""
                    try:
                        std = Cls_dict[replace_avg_with_std(label)]
                        ax.errorbar(Cls[keyword][ell_init:, 0], 
                                    (Cls[keyword][ell_init:, 1] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[1][(map1, map2)][ell_init:],
                                    std[keyword][ell_init:, 1] / Cls_th[1][(map1, map2)][ell_init:],
                                    fmt=".", capsize=3, label=label_plot)
                    except:
                        ax.plot(Cls[keyword][ell_init:, 0], 
                                (Cls[keyword][ell_init:, 1] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[1][(map1, map2)][ell_init:],
                                ".-", label=label_plot)

                ax.set_xscale("log")
                x_limits = ax.get_xlim()
                ax.axvspan(-np.inf, max(ell_min_dict[map1], ell_min_dict[map2]), color="grey", alpha=0.3)
                if ell_max_dict is not None:
                    ax.axvspan(min(ell_max_dict[map1], ell_max_dict[map2]), x_limits[1], color="grey", alpha=0.3)
                ax.set_xlim(x_limits)
                # Determine if this is the last available row in its column
                next_in_col = i + ncols
                if next_in_col >= len(map_list):  # No more plots below in this column
                    ax.set_xlabel(r"$\ell$", fontsize=22)
                else:
                    ax.set_xticklabels([])
                    
                ax.axhspan(-0.05, 0.05, color="grey", alpha=0.1)
                ax.axhline(0, color="k", ls="--")
                ax.set_ylim([-0.2, 0.2])

                if col == 0:
                    if ylabel is None:
                        ax.set_ylabel(r"$C_\ell/C_\ell^{\rm th}-1$", fontsize=22)
                    else:
                        ax.set_ylabel(ylabel, fontsize=22)
                    
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
                ax.set_title(fr"$z_{{\rm sn}}={z_list[i]:.2f},\, z\in[{z_edges_list[i][0]:.2f},{z_edges_list[i][1]:.2f}]$", fontsize=20)

            elif option == 3:
                # Plot normalized difference ((C_l - C_l^th) / sigma)
                for label in list_plot:
                    Cls = Cls_dict[label]
                    if use_keys_for_labels:
                        label_plot = label
                    else:
                        label_parts = []
                        if not N_part_added:
                            label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                        if not N_shells_added:
                            label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                        if not method_added:
                            label_parts.append(fr"{{\rm {label[4]}}}")
                        label_plot = "$" + r",\ ".join(label_parts) + "$"
                    if label_plot == "$$":
                        label_plot = ""
                    try:
                        std = Cls_dict[replace_avg_with_std(label)]
                        ax.errorbar(Cls[keyword][ell_init:, 0], 
                                    (Cls[keyword][ell_init:, 1] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[2][(map1, map2)][ell_init:],
                                    std[keyword][ell_init:, 1] / Cls_th[2][(map1, map2)][ell_init:],
                                    fmt=".", capsize=3, label=label_plot)
                    except:
                        ax.plot(Cls[keyword][ell_init:, 0], 
                                (Cls[keyword][ell_init:, 1] - Cls_th[1][(map1, map2)][ell_init:]) / Cls_th[2][(map1, map2)][ell_init:],
                                ".-", label=label_plot)

                ax.set_xscale("log")
                x_limits = ax.get_xlim()
                ax.axvspan(-np.inf, max(ell_min_dict[map1], ell_min_dict[map2]), color="grey", alpha=0.3)
                if ell_max_dict is not None:
                    ax.axvspan(min(ell_max_dict[map1], ell_max_dict[map2]), x_limits[1], color="grey", alpha=0.3)
                ax.set_xlim(x_limits)
                # Determine if this is the last available row in its column
                next_in_col = i + ncols
                if next_in_col >= len(map_list):  # No more plots below in this column
                    ax.set_xlabel(r"$\ell$", fontsize=22)
                else:
                    ax.set_xticklabels([])

                ax.axhline(0, color="k", ls="--")
                ax.set_ylim([-1, 1])

                if col == 0:
                    if ylabel is None:
                        ax.set_ylabel(r"$(C_\ell-C_\ell^{\rm th})/\sigma_\ell$", fontsize=22)
                    else:
                        ax.set_ylabel(ylabel, fontsize=22)
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
                ax.set_title(fr"$z_{{\rm sn}}={z_list[i]:.2f},\, z\in[{z_edges_list[i][0]:.2f},{z_edges_list[i][1]:.2f}]$", fontsize=20)
                    
    for i in range(len(map_list), len(axes)):
        fig.delaxes(axes[i])
    
    if ncols > 1:
        plt.tight_layout()
    if ncols > 1:
        plt.legend(bbox_to_anchor=(3.2, 0.8), loc="upper right", fontsize=22)
    else:
        plt.legend(bbox_to_anchor=(1, -0.4), loc="upper right", fontsize=22)

    filename = f"plots/Cell_{type_}shells_z{len(map_list)}"
    if filename_extra is not None:
        filename += f"_{filename_extra}"
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")

###########################
###########################
# Plots for shell kappa
###########################
###########################

def plot_xikappa_zsnapshots(map_list, xikappa_dict, xikappa_th, list_plot, use_keys_for_labels=False):
    
    ncols = 5
    nrows = -(-len(map_list) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey="row")

    for i, (map1, map2) in enumerate(zip(map_list, map_list)):
        if map1 == map2:
            row, col = divmod(i, ncols)
            ax = axes[row, col] if nrows > 1 else axes[col]
            keyword = (map1, map2)
            keyword_title = f"({map1+1}, {map2+1})"
            
            try:
                for label in list_plot:
                    xikappa = xikappa_dict[label]
                    if xikappa is not None:
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_plot = fr"$N_{{\rm part}}={label[0]}^3,\ N_{{\rm shells}}={label[3]},\ {{\rm {label[4]}}}$"
                        ax.loglog(
                            xikappa[keyword][:, 0],
                            10**5 * xikappa[keyword][:, 0] * xikappa[keyword][:, 1],
                            ".",
                            label=label_plot
                        )
                        if keyword in xikappa_th:
                            ax.loglog(
                                xikappa_th[keyword][:, 0],
                                10**5 * xikappa_th[keyword][:, 0] * xikappa_th[keyword][:, 1],
                                "-",
                                label="Theory",
                                color="black"
                            )
                x_limits = ax.get_xlim()
                ax.set_xlim(x_limits)
                # Determine if this is the last available row in its column
                next_in_col = i + ncols
                if next_in_col >= len(map_list):  # No more plots below in this column
                    ax.set_xlabel(r"$\theta$ (arcmin)", fontsize=22)
                else:
                    ax.set_xticklabels([])
    
                if col == 0:
                    ax.set_ylabel(r"$10^5\theta\xi_\kappa(\theta)$", fontsize=22)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_title(keyword_title, fontsize=22)

            except:
                ax.remove()
                continue

    for i in range(len(map_list), nrows * ncols):
        row, col = divmod(i, ncols)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(3.2, 0.8), loc="upper right", fontsize=22)
    plt.savefig(f"plots/xikappa_kappashells_z{len(map_list)}.pdf", bbox_inches="tight")

#########################
#########################
# Plots for kappa moments
#########################
#########################

# second moment

def plot_second_moment(option, map_list, second_moment_dict, list_plot, filename_extra=None, use_keys_for_labels=False):

    nbins = len(map_list)
    fig, axes = plt.subplots(nbins, nbins, figsize=(4 * nbins, 3 * nbins), sharey="row")

    title_parts = []

    N_part_added = False
    N_shells_added = False
    method_added = False
    
    if all(t[0] == list_plot[0][0] for t in list_plot):
        title_parts.append(fr"N_{{\rm part}}={list_plot[0][0]}^3")
        N_part_added = True
    if all(t[3] == list_plot[0][3] for t in list_plot):
        title_parts.append(fr"N_{{\rm shells}}={list_plot[0][3]}")
        N_shells_added = True
    if all(t[4] == list_plot[0][4] for t in list_plot):
        title_parts.append(fr"{{\rm {list_plot[0][4]}}}")
        method_added = True
    
    title = "$" + r",\ ".join(title_parts) + "$"

    fig.suptitle(title, fontsize=30)

    nshells_max = max(t[3] for t in list_plot)
    
    num_labels = len(list_plot)
    if option in [2, 3]:
        num_labels -= 1
        colormap = plt.cm.winter_r
    else:
        colormap = plt.cm.cool
    norm = plt.Normalize(0, num_labels - 1)
    values = np.linspace(norm.vmin, norm.vmax, num_labels)
    normalized_values = norm(values)
    # colors = colormap(normalized_values)

    scaled_values = 0.1 + normalized_values * 0.8
    colors = colormap(scaled_values)
    
    for i, map1 in enumerate(map_list):
        for j, map2 in enumerate(map_list):
            if map1 >= map2:
                ax = axes[i, j]
                keyword = f"({map1}, {map2})"
                keyword_title = f"({map1+1}, {map2+1})"
                
                if option == 1:
                    # Plot second moment values
                    for idx, label in enumerate(list_plot):
                        second_moment = second_moment_dict[label]
                        # std = second_moment_dict[replace_avg_with_std(label)]
                        theta_range = list(second_moment.keys())
                        values = []
                        # values_std = []
                        for theta in theta_range:
                            values.append(second_moment[theta][map1, map2])
                            # values_std.append(std[theta][map1, map2])
                        values = np.array(values)
                        # values_std = np.array(values_std)
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                            if not method_added:
                                label_parts.append(fr"{{\rm {label[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                        color = colors[idx]
                        ax.plot(theta_range, values,
                                ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                
                elif option == 2:
                    # Plot percentage differences between labels
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_b[3] == nshells_max:
                                second_moment_a = second_moment_dict[label_a]
                                second_moment_b = second_moment_dict[label_b]
                                std_a = second_moment_dict[replace_avg_with_std(label_a)]
                                std_b = second_moment_dict[replace_avg_with_std(label_b)]
                                theta_range = list(second_moment_a.keys())
                                values_a = []
                                values_b = []
                                values_std_a = []
                                values_std_b = []
                                for theta in theta_range:
                                    values_a.append(second_moment_a[theta][map1, map2])
                                    values_b.append(second_moment_b[theta][map1, map2])
                                    values_std_a.append(std_a[theta][map1, map2])
                                    values_std_b.append(std_b[theta][map1, map2])
                                values_a = np.array(values_a)
                                values_b = np.array(values_b)
                                values_std_a = np.array(values_std_a)
                                values_std_b = np.array(values_std_b)

                                label_parts = []
                                if not N_part_added:
                                    label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                                if not N_shells_added:
                                    label_parts.append(fr"N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]}")
                                else:
                                    print("There is something weird going on, revise the code!")
                                    sys.exit()
                                if not method_added:
                                    label_parts.append(fr"{{\rm {label_a[4]}}}")
                                label_plot = "$" + r",\ ".join(label_parts) + "$"
                                
                                color = colors[a]
                                ax.plot(theta_range, 
                                        # (values_a - values_b) / np.sqrt(values_std_a**2 + values_std_b**2),
                                        (values_a - values_b) / values_std_b,
                                        ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)
                    ax.set_ylim([-1.5, 1.5])

                elif option == 3:
                    # Plot percentage differences between labels
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_b[3] == nshells_max:
                                second_moment_a = second_moment_dict[label_a]
                                second_moment_b = second_moment_dict[label_b]
                                theta_range = list(second_moment_a.keys())
                                values_a = []
                                values_b = []
                                for theta in theta_range:
                                    values_a.append(second_moment_a[theta][map1, map2])
                                    values_b.append(second_moment_b[theta][map1, map2])
                                values_a = np.array(values_a)
                                values_b = np.array(values_b)

                                label_parts = []
                                if not N_part_added:
                                    label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                                if not N_shells_added:
                                    label_parts.append(fr"N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]}")
                                else:
                                    print("There is something weird going on, revise the code!")
                                    sys.exit()
                                if not method_added:
                                    label_parts.append(fr"{{\rm {label_a[4]}}}")
                                label_plot = "$" + r",\ ".join(label_parts) + "$"
                                
                                color = colors[a]
                                ax.plot(theta_range, 
                                        (values_a - values_b) / values_b * 100,
                                        ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-5, 5, color="lightgreen", alpha=0.2)
                    ax.set_ylim([-10, 10])
                
                # Common settings
                if i == nbins - 1:
                    ax.set_xlabel(r"$\vartheta$ (arcmin)", fontsize=22)
                else:
                    ax.set_xticks([])
                if j == 0:
                    ylabel = {
                        1: r"$\langle\kappa^2_\vartheta\rangle$",
                        2: r"$\Delta\langle\kappa^2_\vartheta\rangle/\sigma(\langle\kappa^2_\vartheta\rangle)$",
                        3: r"$\Delta\langle\kappa^2_\vartheta\rangle(\%)$"
                    }[option]
                    ax.set_ylabel(ylabel, fontsize=22)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_title(keyword_title, fontsize=20)
            else:
                axes[i, j].axis("off")
    
    plt.tight_layout()
    plt.subplot(nbins, nbins, 1)
    plt.legend(bbox_to_anchor=(nbins + 0.3, 1), loc="upper right", fontsize=22)

    filename = f"plots/kappa_second_moment_all_{option}"
    if filename_extra is not None:
        filename += f"_{filename_extra}"
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")


# third moment

def plot_third_moment(option, map_list, third_moment_dict, list_plot, filename_extra=None, use_keys_for_labels=False):
    
    nbins = len(map_list)
    fig, axes = plt.subplots(nbins, nbins, figsize=(4 * nbins, 3 * nbins), sharey="row")

    title_parts = []

    N_part_added = False
    N_shells_added = False
    method_added = False
    
    if all(t[0] == list_plot[0][0] for t in list_plot):
        title_parts.append(fr"N_{{\rm part}}={list_plot[0][0]}^3")
        N_part_added = True
    if all(t[3] == list_plot[0][3] for t in list_plot):
        title_parts.append(fr"N_{{\rm shells}}={list_plot[0][3]}")
        N_shells_added = True
    if all(t[4] == list_plot[0][4] for t in list_plot):
        title_parts.append(fr"{{\rm {list_plot[0][4]}}}")
        method_added = True
    
    title = "$" + r",\ ".join(title_parts) + "$"

    fig.suptitle(title, fontsize=30)

    nshells_max = max(t[3] for t in list_plot)
    
    num_labels = len(list_plot)
    if option in [2, 3]:
        num_labels -= 1
        colormap = plt.cm.winter_r
    else:
        colormap = plt.cm.cool
    norm = plt.Normalize(0, num_labels - 1)
    values = np.linspace(norm.vmin, norm.vmax, num_labels)
    normalized_values = norm(values)
    # colors = colormap(normalized_values)

    scaled_values = 0.1 + normalized_values * 0.8
    colors = colormap(scaled_values)
    
    for i, map1 in enumerate(map_list):
        for j, map2 in enumerate(map_list):
            if map1 >= map2:
                ax = axes[i, j]
                keyword = f"({map1}, {map1}, {map2})"
                keyword_title = f"({map1+1}, {map1+1}, {map2+1})"
                
                if option == 1:
                    # Plot third moment values
                    for idx, label in enumerate(list_plot):
                        third_moment = third_moment_dict[label]
                        # std = third_moment_dict[replace_avg_with_std(label)]
                        theta_range = list(third_moment.keys())
                        values = []
                        # values_std = []
                        for theta in theta_range:
                            values.append(third_moment[theta][map1, map1, map2])
                            # values_std.append(std[theta][map1, map1, map2])
                        values = np.array(values)
                        # values_std = np.array(values_std)
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                            if not method_added:
                                label_parts.append(fr"{{\rm {label[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                        color = colors[idx]
                        ax.plot(theta_range, values,
                                    ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                elif option == 2:
                    # Plot percentage differences between labels
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_b[3] == nshells_max:
                                third_moment_a = third_moment_dict[label_a]
                                third_moment_b = third_moment_dict[label_b]
                                std_a = third_moment_dict[replace_avg_with_std(label_a)]
                                std_b = third_moment_dict[replace_avg_with_std(label_b)]
                                theta_range = list(third_moment_a.keys())
                                values_a = []
                                values_b = []
                                values_std_a = []
                                values_std_b = []
                                for theta in theta_range:
                                    values_a.append(third_moment_a[theta][map1, map1, map2])
                                    values_b.append(third_moment_b[theta][map1, map1, map2])
                                    values_std_a.append(std_a[theta][map1, map1, map2])
                                    values_std_b.append(std_b[theta][map1, map1, map2])
                                values_a = np.array(values_a)
                                values_b = np.array(values_b)
                                values_std_a = np.array(values_std_a)
                                values_std_b = np.array(values_std_b)

                                label_parts = []
                                if not N_part_added:
                                    label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                                if not N_shells_added:
                                    label_parts.append(fr"N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]}")
                                else:
                                    print("There is something weird going on, revise the code!")
                                    sys.exit()
                                if not method_added:
                                    label_parts.append(fr"{{\rm {label_a[4]}}}")
                                label_plot = "$" + r",\ ".join(label_parts) + "$"
                                
                                color = colors[a]
                                ax.plot(theta_range, 
                                        # (values_a - values_b) / np.sqrt(values_std_a**2 + values_std_b**2),
                                        (values_a - values_b) / values_std_b,
                                        ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)
                    ax.set_ylim([-1.5, 1.5])

                elif option == 3:
                    # Plot percentage differences between labels
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_b[3] == nshells_max:
                                third_moment_a = third_moment_dict[label_a]
                                third_moment_b = third_moment_dict[label_b]
                                theta_range = list(third_moment_a.keys())
                                values_a = []
                                values_b = []
                                for theta in theta_range:
                                    values_a.append(third_moment_a[theta][map1, map1, map2])
                                    values_b.append(third_moment_b[theta][map1, map1, map2])
                                values_a = np.array(values_a)
                                values_b = np.array(values_b)

                                label_parts = []
                                if not N_part_added:
                                    label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                                if not N_shells_added:
                                    label_parts.append(fr"N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]}")
                                else:
                                    print("There is something weird going on, revise the code!")
                                    sys.exit()
                                if not method_added:
                                    label_parts.append(fr"{{\rm {label_a[4]}}}")
                                label_plot = "$" + r",\ ".join(label_parts) + "$"
                                
                                color = colors[a]
                                ax.plot(theta_range, 
                                        (values_a - values_b) / values_b * 100,
                                        ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-5, 5, color="lightgreen", alpha=0.2)
                    ax.set_ylim([-10, 10])
                    
                # Common settings
                if i == nbins - 1:
                    ax.set_xlabel(r"$\vartheta$ (arcmin)", fontsize=22)
                else:
                    ax.set_xticks([])
                if j == 0:
                    ylabel = {
                        1: r"$\langle\kappa^3_\vartheta\rangle$",
                        2: r"$\Delta\langle\kappa^3_\vartheta\rangle/\sigma(\langle\kappa^3_\vartheta\rangle)$",
                        3: r"$\Delta\langle\kappa^3_\vartheta\rangle(\%)$"
                    }[option]
                    ax.set_ylabel(ylabel, fontsize=22)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_title(keyword_title, fontsize=20)
            else:
                axes[i, j].axis("off")
    
    plt.tight_layout()
    plt.subplot(nbins, nbins, 1)
    plt.legend(bbox_to_anchor=(nbins + 0.3, 1), loc="upper right", fontsize=22)
    filename = f"plots/kappa_third_moment_all_{option}"
    if filename_extra is not None:
        filename += f"_{filename_extra}"
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")

def plot_third_moment_new(option, map_list, third_moment_dict, list_plot, filename_extra=None, use_keys_for_labels=False):
    
    nbins = len(map_list)
    fig, axes = plt.subplots(nbins, nbins, figsize=(4 * nbins, 3 * nbins), sharey="row")

    title_parts = []

    N_part_added = False
    N_shells_added = False
    method_added = False
    
    if all(t[0] == list_plot[0][0] for t in list_plot):
        title_parts.append(fr"N_{{\rm part}}={list_plot[0][0]}^3")
        N_part_added = True
    if all(t[3] == list_plot[0][3] for t in list_plot):
        title_parts.append(fr"N_{{\rm shells}}={list_plot[0][3]}")
        N_shells_added = True
    if all(t[4] == list_plot[0][4] for t in list_plot):
        title_parts.append(fr"{{\rm {list_plot[0][4]}}}")
        method_added = True
    
    title = "$" + r",\ ".join(title_parts) + "$"

    fig.suptitle(title, fontsize=30)

    nshells_max = max(t[3] for t in list_plot)
    
    num_labels = len(list_plot)
    if option in [2, 3]:
        num_labels -= 1
        colormap = plt.cm.winter_r
    else:
        colormap = plt.cm.cool
    norm = plt.Normalize(0, num_labels - 1)
    values = np.linspace(norm.vmin, norm.vmax, num_labels)
    normalized_values = norm(values)
    # colors = colormap(normalized_values)

    scaled_values = 0.1 + normalized_values * 0.8
    colors = colormap(scaled_values)
    
    for i, map1 in enumerate(map_list):
        for j, map2 in enumerate(map_list):
            if map1 >= map2:
                ax = axes[i, j]
                keyword = f"({map1}, {map1}, {map2})"
                keyword_title = f"({map1+1}, {map1+1}, {map2+1})"
                element = tuple([map1, map1, map2])
            else:
                ax = axes[i, j]
                keyword = f"({map2}, {map1}, {map1})"
                keyword_title = f"({map2+1}, {map1+1}, {map1+1})"
                element = tuple([map2, map1, map1])
                
            if option == 1:
                # Plot third moment values
                for idx, label in enumerate(list_plot):
                    third_moment = third_moment_dict[label]
                    # std = third_moment_dict[replace_avg_with_std(label)]
                    theta_range = list(third_moment.keys())
                    values = []
                    # values_std = []
                    for theta in theta_range:
                        values.append(third_moment[theta][element])
                        # values_std.append(std[theta][element])
                    values = np.array(values)
                    # values_std = np.array(values_std)
                    if use_keys_for_labels:
                        label_plot = label
                    else:
                        label_parts = []
                        if not N_part_added:
                            label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                        if not N_shells_added:
                            label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                        if not method_added:
                            label_parts.append(fr"{{\rm {label[4]}}}")
                        label_plot = "$" + r",\ ".join(label_parts) + "$"
                    color = colors[idx]
                    ax.plot(theta_range, values,
                                ".-", label=label_plot, color=color)
                ax.set_xscale("log")
                ax.set_yscale("log")

            elif option == 2:
                # Plot percentage differences between labels
                for (a, label_a) in enumerate(list_plot):
                    for (b, label_b) in enumerate(list_plot):
                        if b > a and label_b[3] == nshells_max:
                            third_moment_a = third_moment_dict[label_a]
                            third_moment_b = third_moment_dict[label_b]
                            std_a = third_moment_dict[replace_avg_with_std(label_a)]
                            std_b = third_moment_dict[replace_avg_with_std(label_b)]
                            theta_range = list(third_moment_a.keys())
                            values_a = []
                            values_b = []
                            values_std_a = []
                            values_std_b = []
                            for theta in theta_range:
                                values_a.append(third_moment_a[theta][element])
                                values_b.append(third_moment_b[theta][element])
                                values_std_a.append(std_a[theta][element])
                                values_std_b.append(std_b[theta][element])
                            values_a = np.array(values_a)
                            values_b = np.array(values_b)
                            values_std_a = np.array(values_std_a)
                            values_std_b = np.array(values_std_b)

                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]}")
                            else:
                                print("There is something weird going on, revise the code!")
                                sys.exit()
                            if not method_added:
                                label_parts.append(fr"{{\rm {label_a[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                            
                            color = colors[a]
                            ax.plot(theta_range, 
                                    # (values_a - values_b) / np.sqrt(values_std_a**2 + values_std_b**2),
                                    (values_a - values_b) / values_std_b,
                                    ".-", label=label_plot, color=color)
                ax.set_xscale("log")
                ax.axhline(0, color="k", ls="--")
                ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)
                ax.set_ylim([-1.5, 1.5])

            elif option == 3:
                # Plot percentage differences between labels
                for (a, label_a) in enumerate(list_plot):
                    for (b, label_b) in enumerate(list_plot):
                        if b > a and label_b[3] == nshells_max:
                            third_moment_a = third_moment_dict[label_a]
                            third_moment_b = third_moment_dict[label_b]
                            theta_range = list(third_moment_a.keys())
                            values_a = []
                            values_b = []
                            for theta in theta_range:
                                values_a.append(third_moment_a[theta][element])
                                values_b.append(third_moment_b[theta][element])
                            values_a = np.array(values_a)
                            values_b = np.array(values_b)

                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]}")
                            else:
                                print("There is something weird going on, revise the code!")
                                sys.exit()
                            if not method_added:
                                label_parts.append(fr"{{\rm {label_a[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                            
                            color = colors[a]
                            ax.plot(theta_range, 
                                    (values_a - values_b) / values_b * 100,
                                    ".-", label=label_plot, color=color)
                ax.set_xscale("log")
                ax.axhline(0, color="k", ls="--")
                ax.axhspan(-5, 5, color="lightgreen", alpha=0.2)
                ax.set_ylim([-10, 10])
                
            # Common settings
            if i == nbins - 1:
                ax.set_xlabel(r"$\vartheta$ (arcmin)", fontsize=22)
            else:
                ax.set_xticks([])
            if j == 0:
                ylabel = {
                    1: r"$\langle\kappa^3_\vartheta\rangle$",
                    2: r"$\Delta\langle\kappa^3_\vartheta\rangle/\sigma(\langle\kappa^3_\vartheta\rangle)$",
                    3: r"$\Delta\langle\kappa^3_\vartheta\rangle(\%)$"
                }[option]
                ax.set_ylabel(ylabel, fontsize=22)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_title(keyword_title, fontsize=20)
    
    plt.tight_layout()
    # plt.subplot(nbins, nbins, 1)
    # plt.legend(bbox_to_anchor=(nbins + 0.3, 1), loc="upper right", fontsize=22)
    filename = f"plots/kappa_third_moment_all_{option}"
    if filename_extra is not None:
        filename += f"_{filename_extra}"
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    
# fourth moment

def plot_fourth_moment(option, map_list, fourth_moment_dict, list_plot, filename_extra=None, use_keys_for_labels=False):
    
    nbins = len(map_list)
    fig, axes = plt.subplots(nbins, nbins, figsize=(4 * nbins, 3 * nbins), sharey="row")

    title_parts = []

    N_part_added = False
    N_shells_added = False
    method_added = False
    
    if all(t[0] == list_plot[0][0] for t in list_plot):
        title_parts.append(fr"N_{{\rm part}}={list_plot[0][0]}^3")
        N_part_added = True
    if all(t[3] == list_plot[0][3] for t in list_plot):
        title_parts.append(fr"N_{{\rm shells}}={list_plot[0][3]}")
        N_shells_added = True
    if all(t[4] == list_plot[0][4] for t in list_plot):
        title_parts.append(fr"{{\rm {list_plot[0][4]}}}")
        method_added = True
    
    title = "$" + r",\ ".join(title_parts) + "$"

    nshells_max = max(t[3] for t in list_plot)

    fig.suptitle(title, fontsize=30)
    
    num_labels = len(list_plot)
    if option in [2, 3]:
        num_labels -= 1
        colormap = plt.cm.winter_r
    else:
        colormap = plt.cm.cool
    norm = plt.Normalize(0, num_labels - 1)
    values = np.linspace(norm.vmin, norm.vmax, num_labels)
    normalized_values = norm(values)
    # colors = colormap(normalized_values)

    scaled_values = 0.1 + normalized_values * 0.8
    colors = colormap(scaled_values)
    
    for i, map1 in enumerate(map_list):
        for j, map2 in enumerate(map_list):
            if map1 >= map2:
                ax = axes[i, j]
                keyword = f"({map1}, {map1}, {map2}, {map2})"
                keyword_title = f"({map1+1}, {map1+1}, {map2+1}, {map2+1})"
                
                if option == 1:
                    # Plot fourth moment values
                    for idx, label in enumerate(list_plot):
                        fourth_moment = fourth_moment_dict[label]
                        # std = fourth_moment_dict[replace_avg_with_std(label)]
                        theta_range = list(fourth_moment.keys())
                        values = []
                        # values_std = []
                        for theta in theta_range:
                            values.append(fourth_moment[theta][map1, map1, map2, map2])
                            # values_std.append(std[theta][map1, map1, map2,  map2])
                        values = np.array(values)
                        # values_std = np.array(values_std)
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_parts = []
                            if not N_part_added:
                                label_parts.append(fr"N_{{\rm part}}={label[0]}^3")
                            if not N_shells_added:
                                label_parts.append(fr"N_{{\rm shells}}={label[3]}")
                            if not method_added:
                                label_parts.append(fr"{{\rm {label[4]}}}")
                            label_plot = "$" + r",\ ".join(label_parts) + "$"
                        color = colors[idx]
                        # ax.errorbar(theta_range, values, values_std,
                        #             fmt=".", capsize=3, label=label_plot, color=color)
                        ax.plot(theta_range, values,
                                    ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                
                elif option == 2:
                    # Plot percentage differences between labels
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_b[3] == nshells_max:
                                fourth_moment_a = fourth_moment_dict[label_a]
                                fourth_moment_b = fourth_moment_dict[label_b]
                                std_a = fourth_moment_dict[replace_avg_with_std(label_a)]
                                std_b = fourth_moment_dict[replace_avg_with_std(label_b)]
                                theta_range = list(fourth_moment_a.keys())
                                values_a = []
                                values_b = []
                                values_std_a = []
                                values_std_b = []
                                for theta in theta_range:
                                    values_a.append(fourth_moment_a[theta][map1, map1, map2, map2])
                                    values_b.append(fourth_moment_b[theta][map1, map1, map2, map2])
                                    values_std_a.append(std_a[theta][map1, map1, map2, map2])
                                    values_std_b.append(std_b[theta][map1, map1, map2, map2])
                                values_a = np.array(values_a)
                                values_b = np.array(values_b)
                                values_std_a = np.array(values_std_a)
                                values_std_b = np.array(values_std_b)

                                label_parts = []
                                if not N_part_added:
                                    label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                                if not N_shells_added:
                                    label_parts.append(fr"N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]}")
                                else:
                                    print("There is something weird going on, revise the code!")
                                    sys.exit()
                                if not method_added:
                                    label_parts.append(fr"{{\rm {label_a[4]}}}")
                                label_plot = "$" + r",\ ".join(label_parts) + "$"
                                
                                color = colors[a]
                                ax.plot(theta_range, 
                                        # (values_a - values_b) / np.sqrt(values_std_a**2 + values_std_b**2),
                                        (values_a - values_b) / values_std_b,
                                        ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)
                    ax.set_ylim([-1.5, 1.5])

                elif option == 3:
                    # Plot percentage differences between labels
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_b[3] == nshells_max:
                                fourth_moment_a = fourth_moment_dict[label_a]
                                fourth_moment_b = fourth_moment_dict[label_b]
                                theta_range = list(fourth_moment_a.keys())
                                values_a = []
                                values_b = []
                                for theta in theta_range:
                                    values_a.append(fourth_moment_a[theta][map1, map1, map2, map2])
                                    values_b.append(fourth_moment_b[theta][map1, map1, map2, map2])
                                values_a = np.array(values_a)
                                values_b = np.array(values_b)

                                label_parts = []
                                if not N_part_added:
                                    label_parts.append(fr"N_{{\rm part}}={label_a[0]}^3")
                                if not N_shells_added:
                                    label_parts.append(fr"N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]}")
                                else:
                                    print("There is something weird going on, revise the code!")
                                    sys.exit()
                                if not method_added:
                                    label_parts.append(fr"{{\rm {label_a[4]}}}")
                                label_plot = "$" + r",\ ".join(label_parts) + "$"
                                
                                color = colors[a]
                                ax.plot(theta_range, 
                                        (values_a - values_b) / values_b * 100,
                                        ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-5, 5, color="lightgreen", alpha=0.2)
                    ax.set_ylim([-10, 10])

                # Common settings
                if i == nbins - 1:
                    ax.set_xlabel(r"$\vartheta$ (arcmin)", fontsize=22)
                else:
                    ax.set_xticks([])
                if j == 0:
                    ylabel = {
                        1: r"$\langle\kappa^4_\vartheta\rangle$",
                        2: r"$\Delta\langle\kappa^4_\vartheta\rangle/\sigma(\langle\kappa^4_\vartheta\rangle)$",
                        3: r"$\Delta\langle\kappa^4_\vartheta\rangle(\%)$"
                    }[option]
                    ax.set_ylabel(ylabel, fontsize=22)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_title(keyword_title, fontsize=20)
            else:
                axes[i, j].axis("off")
    
    plt.tight_layout()
    plt.subplot(nbins, nbins, 1)
    plt.legend(bbox_to_anchor=(nbins + 0.3, 1), loc="upper right", fontsize=22)
    filename = f"plots/kappa_fourth_moment_all_{option}"
    if filename_extra is not None:
        filename += f"_{filename_extra}"
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")

###########################
###########################
# Plots for kappa xipm
###########################
###########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_xipm_kappa(option, mode, map_list, xipm_dict, xipm_th, xipm_err, list_plot, use_keys_for_labels=False):
    
    if option == 2:
        raise NotImplementedError("Option 2 is not implemented for this function.")

    if mode not in ["xip", "xim", "xikappa"]:
        raise ValueError("Mode must be 'xip', 'xim' or 'xikappa'.")

    ylabel_map = {
        "xip": {
            1: r"$10^5\theta\xi_+(\theta)$",
            3: r"$(\xi_+ - \xi_+^{\rm th})(\theta) / \sigma_+(\theta)$",
            4: r"$\Delta \xi_+ / \sigma_+ (\theta)$"
        },
        "xim": {
            1: r"$10^5\theta\xi_-(\theta)$",
            3: r"$(\xi_- - \xi_-^{\rm th})(\theta) / \sigma_-(\theta)$",
            4: r"$\Delta \xi_- / \sigma_- (\theta)$"
        },
        "xikappa": {
            1: r"$10^5\theta\xi_\kappa(\theta)$",
            3: r"$(\xi_\kappa - \xi_\kappa^{\rm th})(\theta) / \sigma_\kappa(\theta)$",
            4: r"$\Delta \xi_\kappa / \sigma_\kappa (\theta)$"
        }
    }

    nbins = len(map_list)
    fig, axes = plt.subplots(nbins, nbins, figsize=(4 * nbins, 3 * nbins), sharey="row")

    num_labels = len(list_plot)
    if option in [4]:
        num_labels -= 1
        colormap = plt.cm.winter_r
    else:
        colormap = plt.cm.cool
    norm = plt.Normalize(0, num_labels - 1)
    values = np.linspace(norm.vmin, norm.vmax, num_labels)
    normalized_values = norm(values)
    # colors = colormap(normalized_values)

    scaled_values = 0.1 + normalized_values * 0.8
    colors = colormap(scaled_values)
    
    for i, map1 in enumerate(map_list):
        for j, map2 in enumerate(map_list):
            if map1 >= map2:
                ax = axes[i, j]
                keyword = (map1, map2)
                keyword_title = (map1+1, map2+1)

                if option == 1:
                    # Plot xi+ or xi- with error bars and theory line
                    for idx, label in enumerate(list_plot):
                        xipm = xipm_dict[label]
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_plot = fr"$N_{{\rm part}}={label[0]}^3,\ N_{{\rm shells}}={label[3]},\ {{\rm {label[4]}}}$"
                        color = colors[idx]
                        ax.errorbar(xipm[keyword][:, 0], 
                                    10**5 * xipm[keyword][:, 0] * xipm[keyword][:, 1], 
                                    yerr=10**5 * xipm[keyword][:, 0] * xipm_err[keyword], 
                                    fmt=".", capsize=3, label=label_plot, color=color)
                    ax.semilogx(xipm_th[keyword][:, 0], 
                                10**5 * xipm_th[keyword][:, 0] * xipm_th[keyword][:, 1], 
                                "-", label="Theory", color="black")

                elif option == 3:
                    # Plot normalized residuals between xi+ or xi- and theory
                    for idx, label in enumerate(list_plot):
                        xipm = xipm_dict[label]
                        if "avg" in label:
                            std = xipm_dict[replace_avg_with_std(label)]
                        else:
                            std = None
                        if use_keys_for_labels:
                            label_plot = label
                        else:
                            label_plot = fr"$N_{{\rm part}}={label[0]}^3,\ N_{{\rm shells}}={label[3]},\ {{\rm {label[4]}}}$"
                        color = colors[idx]
                        if std is not None:
                            ax.errorbar(xipm[keyword][:, 0], 
                                        (xipm[keyword][:, 1] - xipm_th[keyword][:, 1]) / xipm_err[keyword],
                                        std[keyword][:, 1] / xipm_err[keyword],
                                        fmt=".", capsize=3, label=label_plot, color=color)
                        else:
                            ax.plot(xipm[keyword][:, 0], 
                                    (xipm[keyword][:, 1] - xipm_th[keyword][:, 1]) / xipm_err[keyword],
                                    ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)

                elif option == 4:
                    # Plot relative differences between xi+ or xi- datasets
                    for (a, label_a) in enumerate(list_plot):
                        for (b, label_b) in enumerate(list_plot):
                            if b > a and label_b[3] == 101:
                                xipm_a = xipm_dict[label_a]
                                xipm_b = xipm_dict[label_b]
                                if "avg" in label_a and "avg" in label_b:
                                    std_a = xipm_dict[replace_avg_with_std(label_a)]
                                    std_b = xipm_dict[replace_avg_with_std(label_b)]
                                else:
                                    std_a = None
                                label_plot = fr"$N_{{\rm part}}={label_a[0]}^3,\ N_{{\rm shells}}={label_a[3]}\ {{\rm vs.}}\ {label_b[3]},\ {{\rm {label_a[4]}}}$"
                                color = colors[a]
                                if std_a is not None:
                                    ax.errorbar(xipm_a[keyword][:, 0], 
                                                (xipm_a[keyword][:, 1] - xipm_b[keyword][:, 1]) / xipm_err[keyword],
                                                np.sqrt(std_a[keyword][:, 1]**2 + std_b[keyword][:, 1]**2) / xipm_err[keyword],
                                                fmt=".", capsize=3, label=label_plot, color=color)
                                else:
                                    ax.plot(xipm_a[keyword][:, 0], 
                                            (xipm_a[keyword][:, 1] - xipm_b[keyword][:, 1]) / xipm_err[keyword],
                                            ".-", label=label_plot, color=color)
                    ax.set_xscale("log")
                    ax.axhline(0, color="k", ls="--")
                    ax.axhspan(-1, 1, color="#FFB84D", alpha=0.1)

                # Common settings
                if i == nbins - 1:
                    ax.set_xlabel(r"$\theta$ (arcmin)", fontsize=22)
                else:
                    ax.set_xticks([])
                if j == 0:
                    ax.set_ylabel(ylabel_map[mode][option], fontsize=22)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                ax.set_title(f"{keyword_title}", fontsize=20)
            else:
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.subplot(nbins, nbins, 1)
    plt.legend(bbox_to_anchor=(nbins + 0.3, 1), loc="upper right", fontsize=22)
    filename = f"plots/{mode}_all_{option}"
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")

###########################
###########################
# Plots for w(theta)
###########################
###########################

# def plot_wtheta(map_list, wtheta_dict, wtheta_th, wtheta_err, use_keys_for_labels=False):
def plot_wtheta(map_list, wtheta_dict, wtheta_th, list_plot, use_keys_for_labels=False):

    fig, axes = plt.subplots(1, len(map_list), figsize=(6 * len(map_list), 4))

    num_labels = len(list_plot)
    colormap = plt.cm.cool
    norm = plt.Normalize(0, num_labels - 1)
    values = np.linspace(norm.vmin, norm.vmax, num_labels)
    normalized_values = norm(values)
    # colors = colormap(normalized_values)

    scaled_values = 0.1 + normalized_values * 0.8
    colors = colormap(scaled_values)

    for i, map1 in enumerate(map_list):
        ax = axes[i]
        keyword = map1
        keyword_title = map1+1

        for idx, label in enumerate(list_plot):
            wtheta = wtheta_dict[label]
            if "avg" in label:
                std = wtheta_dict[replace_avg_with_std(label)]
            else:
                std = None
            if use_keys_for_labels:
                label_plot = label
            else:
                # label_plot = fr"$N_{{\rm part}}={label[0]}^3,\ {{\rm sim}}={{\rm {label[1]}}},\ {{\rm seed}}={{\rm {label[2]}}},\ N_{{\rm shells}}={label[3]},\ {{\rm {label[4]}}}$"
                label_plot = fr"$N_{{\rm part}}={label[0]}^3,\ N_{{\rm shells}}={label[3]},\ {{\rm {label[4]}}}$"
            color = colors[idx]
            if std is not None:
                ax.errorbar(wtheta[keyword][:, 0] / 60, 100 * (wtheta[keyword][:, 0] / 60) * wtheta[keyword][:, 1], yerr=100 * (wtheta[keyword][:, 0] / 60) * std[keyword][:, 1], fmt=".", capsize=3, label=label_plot, color=color)
            else:
                ax.plot(wtheta[keyword][:, 0] / 60, 100 * (wtheta[keyword][:, 0] / 60) * wtheta[keyword][:, 1], ".-", label=label_plot, color=color)
            # ax.plot(wtheta[keyword][:, 0] / 60, 100 * (wtheta[keyword][:, 0] / 60) * wtheta[keyword][:, 1], ".-", label=label_plot)

        ax.semilogx(wtheta_th[keyword][:, 0] / 60, 100 * (wtheta_th[keyword][:, 0] / 60) * wtheta_th[keyword][:, 1], "-", label="Theory", color="black")

        ax.set_xlabel(r"$\theta$ (deg)", fontsize=22)
        if i == 0:
            ax.set_ylabel(r"$10^2\theta w(\theta)$", fontsize=22)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_title(f"({keyword_title})", fontsize=20)

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=22)

    plt.tight_layout()
    plt.savefig(f"plots/wtheta_all.pdf", bbox_inches="tight")
    