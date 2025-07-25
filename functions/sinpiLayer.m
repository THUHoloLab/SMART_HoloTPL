classdef sinpiLayer < nnet.layer.Layer
 
    methods
        function layer = sinpiLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "pi*sin of argument";
            
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            Z = pi*sin(X);
        end
        
    end
end