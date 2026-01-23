import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles


def two_rules_redundancy(
    rule1_total, rule1_bad,
    rule2_total, rule2_bad,
    rule12_both_total, rule12_both_bad,
    rule1_name, rule2_name
):
    only_r1_total = rule1_total - rule12_both_total
    only_r2_total = rule2_total - rule12_both_total

    venn2(
        subsets=(only_r1_total, only_r2_total, rule12_both_total),
        set_labels=(rule1_name, rule2_name),
        set_colors=('orange', 'blue'),
        alpha=0.6
    )
    venn2_circles(
        subsets=(only_r1_total, only_r2_total, rule12_both_total),
        linestyle='dashed',
        linewidth=1
    )
    plt.title('All Declined Accounts')
    plt.show()

    only_r1_bad = rule1_bad - rule12_both_bad
    only_r2_bad = rule2_bad - rule12_both_bad

    venn2(
        subsets=(only_r1_bad, only_r2_bad, rule12_both_bad),
        set_labels=(rule1_name, rule2_name),
        set_colors=('orange', 'blue'),
        alpha=0.6
    )
    venn2_circles(
        subsets=(only_r1_bad, only_r2_bad, rule12_both_bad),
        linestyle='dashed',
        linewidth=1
    )
    plt.title('Detected Bad Accounts')
    plt.show()


def three_rules_redundancy(
    r1_total, r1_bad,
    r2_total, r2_bad,
    r3_total, r3_bad,
    r12_total, r12_bad,
    r13_total, r13_bad,
    r23_total, r23_bad,
    r123_total, r123_bad,
    rule1_name, rule2_name, rule3_name
):
    only_r1 = r1_total - r12_total - r13_total + r123_total
    only_r2 = r2_total - r12_total - r23_total + r123_total
    only_r3 = r3_total - r13_total - r23_total + r123_total

    only_r12 = r12_total - r123_total
    only_r13 = r13_total - r123_total
    only_r23 = r23_total - r123_total
    only_r123 = r123_total

    venn3(
        subsets=(only_r1, only_r2, only_r12, only_r3, only_r13, only_r23, only_r123),
        set_labels=(rule1_name, rule2_name, rule3_name),
        set_colors=('orange', 'blue', 'green'),
        alpha=0.6
    )
    venn3_circles(
        subsets=(only_r1, only_r2, only_r12, only_r3, only_r13, only_r23, only_r123),
        linestyle='dashed',
        linewidth=1
    )
    plt.title("All Declined Accounts")
    plt.show()

    only_r1_bad = r1_bad - r12_bad - r13_bad + r123_bad
    only_r2_bad = r2_bad - r12_bad - r23_bad + r123_bad
    only_r3_bad = r3_bad - r13_bad - r23_bad + r123_bad

    only_r12_bad = r12_bad - r123_bad
    only_r13_bad = r13_bad - r123_bad
    only_r23_bad = r23_bad - r123_bad
    only_r123_bad = r123_bad

    venn3(
        subsets=(only_r1_bad, only_r2_bad, only_r12_bad,
                 only_r3_bad, only_r13_bad, only_r23_bad, only_r123_bad),
        set_labels=(rule1_name, rule2_name, rule3_name),
        set_colors=('orange', 'blue', 'green'),
        alpha=0.6
    )
    venn3_circles(
        subsets=(only_r1_bad, only_r2_bad, only_r12_bad,
                 only_r3_bad, only_r13_bad, only_r23_bad, only_r123_bad),
        linestyle='dashed',
        linewidth=1
    )
    plt.title("Detected Bad Accounts")
    plt.show()
