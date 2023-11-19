figure, imshow(x)
title(['Mondedas 10 centavos: ',num2str(nummonedachica),...
'Mondedas 1 peso: ', num2str(nummonedagrande1),...
'Monedas 50 centavos: ', num2str(nummonedagrande2),...
'Dados: ',num2str(numdados)],'FontSize',16)
hold on
plot(centroids(:,1),centroids(:,2),'r*')
for k=1:length(innddados)
    text(centroids(inddados(k),1)+100,...
    centroids(inddados(k),2),['D',num2str(numd{k})],...
    'color','g','fontsize',16,'fontweight','bold');
end
