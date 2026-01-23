from plotly.subplots import make_subplots
import plotly.graph_objects as go


def group_performance_one_rule(
    group_data,
    top_x,
    original_bal,
    new_bal,
    original_bad_bal,
    new_bad_bal,
    original_bal_br,
    new_bal_br
):
    selected = group_data[group_data.index != 'Missing'].head(top_x)

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(go.Bar(x=selected.index, y=selected[original_bad_bal], name='Original Bad Bal'))
    fig.add_trace(go.Bar(x=selected.index, y=selected[new_bad_bal], name='New Bad Bal'))
    fig.add_trace(go.Scatter(x=selected.index, y=selected[original_bal_br],
                             mode='lines+markers', name='Original BR'),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br],
                             mode='lines+markers', name='New BR'),
                  secondary_y=True)

    fig.update_layout(
        title='Bad Balance & BR Changes',
        barmode='group',
        template='plotly_white'
    )
    fig.show()


def group_performance_two_rules(
    group_data,
    top_x,
    original_bal,
    new_bal1,
    new_bal2,
    original_bad_bal,
    new_bad_bal1,
    new_bad_bal2,
    original_bal_br,
    new_bal_br1,
    new_bal_br2
):
    selected = group_data[group_data.index != 'Missing'].head(top_x)

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(go.Bar(x=selected.index, y=selected[original_bad_bal], name='Original'))
    fig.add_trace(go.Bar(x=selected.index, y=selected[new_bad_bal1], name='Rule 1'))
    fig.add_trace(go.Bar(x=selected.index, y=selected[new_bad_bal2], name='Rule 2'))

    fig.add_trace(go.Scatter(x=selected.index, y=selected[original_bal_br],
                             mode='lines+markers', name='Original BR'),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br1],
                             mode='lines+markers', name='Rule 1 BR'),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br2],
                             mode='lines+markers', name='Rule 2 BR'),
                  secondary_y=True)

    fig.update_layout(
        title='Balance & BR Comparison',
        barmode='group',
        template='plotly_white'
    )
    fig.show()
