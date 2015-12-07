clear all
clc

ImageRead = imread('1.png');
X = zeros((size(ImageRead,1)*size(ImageRead,2)/64), 64);
k=0;
for i=1:8:size(ImageRead,1)
    for j=1:8:size(ImageRead,1)
       one_block_values= ImageRead(i:i+8-1,j:j+8-1);
       k=k+1;
       X(k,:)=reshape(one_block_values',1,64);
    end
end
mean_x=mean(X);
for i=1:64
	X(:,i)=X(:,i)-mean_x(i);
end
co=zeros(64);
for i=1:64
    for j=1:64 
       co(i,j)=sum(X(:,i).*X(:,j))/(size(ImageRead,1)*size(ImageRead,2)/64);
    end
end
[V,D]=eig(co);
for s=[5 10 20 30 40 50 60]
    flippedV=fliplr(V);
    standardizedX=flippedV'/norm(flippedV')*X';
    flippedV(:,64-s+1:end)=0;
    retrivedX=(flippedV/norm(flippedV))*standardizedX;
    retrivedX=retrivedX';
    for i=1:64
		retrivedX(:,i)=retrivedX(:,i)+mean_x(i);
    end
    FinalImage=zeros(size(ImageRead,1),size(ImageRead,2));
    k=1;
    for i=1:8:size(ImageRead,1)
        for j=1:8:size(ImageRead,1)
            FinalImage(i:i+8-1,j:j+8-1)=reshape(retrivedX(k,:),8,8)';
            k=k+1;
        end
    end
    figure
    imshow(FinalImage,[]);
	print(['converted/',num2str(s)],'-dpng')
end