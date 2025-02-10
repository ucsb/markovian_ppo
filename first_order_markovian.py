def get_first_order_markovian_score(self,
                                        p_matrix,
                                        val_matrix,
                                        conf_matrix=None,
                                        graph=None,
                                        ambiguous_triples=None,
                                        alpha_level=0.05):
        """
        Returns the first-order Markovian score for the given p_matrix and val_matrix.
        """
        if graph is not None:
            sig_links = (graph != "")*(graph != "<--")
        else:
            sig_links = (p_matrix <= alpha_level)

        print("\n## Significant links at alpha = %s:" % alpha_level)
        total_violation = 0.0
        for j in range(self.N):
            links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                     for p in zip(*np.where(sig_links[:, j, :]))}
            # Sort by value
            sorted_links = sorted(links, key=links.get, reverse=True)
            n_links = len(links)
            string = ("\n    Variable %s has %d "
                      "link(s):" % (self.var_names[j], n_links))
            for p in sorted_links:
                string += ("\n        (%s % d): pval = %.5f" %
                           (self.var_names[p[0]], p[1],
                            p_matrix[p[0], j, abs(p[1])]))
                string += " | val = % .3f" % (
                    val_matrix[p[0], j, abs(p[1])])
                if p[1] < -1:
                    total_violation += abs(val_matrix[p[0], j, abs(p[1])])
                if conf_matrix is not None:
                    string += " | conf = (%.3f, %.3f)" % (
                        conf_matrix[p[0], j, abs(p[1])][0],
                        conf_matrix[p[0], j, abs(p[1])][1])
                if graph is not None:
                    if p[1] == 0 and graph[j, p[0], 0] == "o-o":
                        string += " | unoriented link"
                    if graph[p[0], j, abs(p[1])] == "x-x":
                        string += " | unclear orientation due to conflict"

        # link_marker = {True:"o-o", False:"-->"}

        if ambiguous_triples is not None and len(ambiguous_triples) > 0:
            print("\n## Ambiguous triples (not used for orientation):\n")
            for triple in ambiguous_triples:
                (i, tau), k, j = triple
                print("    [(%s % d), %s, %s]" % (
                    self.var_names[i], tau, 
                    self.var_names[k],
                    self.var_names[j]))
        return total_violation