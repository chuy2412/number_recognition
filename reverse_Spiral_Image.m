%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%reverse_Spiral_Image
%takes a spiral image and saves the original version
%of that image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = reverse_Spiral_Image(Z)
    pixels = 28*28;
    fdim = [28 28];
    X= zeros(28,28);
    count = 1;
    table = zeros(pixels,1);
    
    %Start = center (r/2,c/2)
    row = 14;
    col = 14;
    
    X(row,col) = Z(count);
    table(row,col) = 1;    %mark as visited
    count =count +1;
    
    while (count <=pixels)
        %Go right
        [row,col] = move_right(row,col);
        X(row,col) = Z(count);
        table(row,col) = 1; %mark as visited
        count = count + 1;
        
        %While (down is visited)
        while (is_down_visited(table,row,col))
            %Go right
            [row,col] = move_right(row,col);
            X(row,col) = Z(count);
            table(row,col) = 1; %mark as visited 
            count = count+1;
        end

        %Go down
        [row,col] = move_down(row,col);
        X(row,col) = Z(count);
        table(row,col) = 1; %mark as visited 
        count = count+1;
        
        %While (left is visited)
        while (is_left_visited(table,row,col))
            %Go down
            [row,col] = move_down(row,col);
            X(row,col) = Z(count);
            table(row,col) = 1; %mark as visited 
            count = count+1;
        end
    
        %Go left
        [row,col] = move_left(row,col);
        X(row,col) = Z(count);
        table(row,col) = 1; %mark as visited 
        count = count+1;
        
        %While up is visited
        while ((is_up_visited(table,row,col) && count<=pixels))
            if count <=pixels
                %Go left
                [row,col] = move_left(row,col);
                X(row,col) = Z(count);
                table(row,col) = 1; %mark as visited 
                count = count+1;
            end
        end
    
        if count <=pixels
            %Go up
            [row,col] = move_up(row,col);
            X(row,col) = Z(count);
            table(row,col) = 1; %mark as visited 
            count = count+1;
        end
        
        %While right is visited
        while ((is_right_visited(table,row,col)) && (count <= pixels))
            %Go up
            [row,col] = move_up(row,col);
            X(row,col) = Z(count);
            table(row,col) = 1; %mark as visited 
            count = count+1;
        end
    end
    
    X = uint8(X);
    
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%move_up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [row, col] = move_up(r,c)
    col = c;
    if (r ~= 1)
        row = r -1;
    else
        row = r;
    end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%is_up_visited
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = is_up_visited(table,r,c)
    [row,col] = move_up(r,c);
    result = table(row,col);
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%move_down
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [row, col] = move_down(r,c)
    col = c;
    if (r ~= 28)
        row = r + 1;
    else
        row = r;
    end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%is_down_visited
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = is_down_visited(table,r,c)
    [row,col] = move_down(r,c);
    result = table(row,col);
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%move_left
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [row, col] = move_left(r,c)
    row = r;
    if (c ~= 1)
        col = c -1;
    else
        col = c;
    end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%is_left_visited
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = is_left_visited(table,r,c)
    [row,col] = move_left(r,c);
    result = table(row,col);
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%move_right
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [row, col] = move_right(r,c)
    row = r;
    if (c ~= 28)
        col = c + 1;
    else
        col = c;
    end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%is_right_visited
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = is_right_visited(table,r,c)
    [row,col] = move_right(r,c);
    result = table(row,col);
return