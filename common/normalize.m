function x = normalize(y)
%% Normalizes each column such that the norm of each col is 1
%
% Created by Stephen Tierney
% stierney@csu.edu.au
%

    for i=1:size(y,2)
        x(:,i) = y(:,i)/sqrt(y(:,i)'*y(:,i));
    end
end

%     for i=1:size(y,2)
%         for k = 1:3
%         x(:,i,k) = y(:,i,k)/sqrt(y(:,i,k)'*y(:,i,k));
%         end
%     end
% end
% clear,clc
% a=imread('4.png');
% imshow(a)
% r=a(:,:,1);
% g=a(:,:,2);
% b=a(:,:,3);
% r1=cast(r,'double')/.255;
% g1=cast(g,'double')/.255;
% b1=cast(b,'double')/.255;
% c=cat(3,r1,g1,b1);
% figure,imshow(uint8(c))
