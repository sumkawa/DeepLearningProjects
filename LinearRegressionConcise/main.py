from LinearRegression import LinearRegression
import d2l
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensore([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)