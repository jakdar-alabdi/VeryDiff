sysimage_dir = @__DIR__

open("./test.log", "w") do f
    redirect_stdout(f) do 
        redirect_stderr(f) do 
            println("Hello World!")
        end
    end
end

