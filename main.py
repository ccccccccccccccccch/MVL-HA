from model import main

epochs = [50]
finance_units = [30, 60]
stock_units = [30, 60]
network_units = [30, 60]
attention_units = [30]
mlp_units = [20]
finance_drops = [0.3, 0.5]
stock_drops = [0.3, 0.5]
network_drops = [0.3, 0.5]

main(epochs=epochs,
     finance_units=finance_units,
     stock_units=stock_units,
     network_units=network_units,
     attention_units=attention_units,
     mlp_units=mlp_units,
     finance_drops=finance_drops,
     stock_drops=stock_drops,
     network_drops=network_drops)
