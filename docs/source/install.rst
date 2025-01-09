.. _install:

============
Installation
============

The **rust_nurbs** library can be installed any of the following methods:

.. tab-set::

    .. tab-item:: Stable

        .. code-block:: shell

            pip install rust-nurbs
    
    .. tab-item:: Latest

        .. code-block:: shell

            git clone https://github.com/mlau154/rust_nurbs.git
            cd rust_nurbs
            pip install .[build]
    
    .. tab-item:: Dev

        .. code-block:: shell

            git clone https://github.com/mlau154/rust_nurbs.git
            cd rust_nurbs
            pip install .[dev]

.. important::

    If you do not want ``rust-nurbs`` installed in your default system Python, make sure to first activate a virtual environment!

You may find that the ``Latest`` and ``Dev`` methods of installing **rust_nurbs** fail because Rust/Cargo could not be found on your system. Rust must be installed properly on your machine for these versions of the ``pip install`` command to pass. See below to
see how to ensure this in Windows or Linux (macOS likely has similar instructions to the Linux case):

.. tab-set::

    .. tab-item:: Windows (VS C++)

        #. Install the `Visual Studio C++ Build toolchain <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.
        #. Add the VS C++ toolchain to your `system path environment variable <https://www.eukhost.com/kb/how-to-add-to-the-path-on-windows-10-and-windows-11/>`_ (if installed at the default location, the path to the binaries will usually look something like ``C:\Program Files (x86)\Microsoft Visual Studio\<release-year>\BuildTools\MSBuild\Current\Bin`` or ``C:\Program Files\Microsoft Visual Studio\<release-year>\BuildTools\MSBuild\Current\Bin`` if using the 64-bit version)
        #. Install Rust from the `Rust installation page <https://www.rust-lang.org/learn/get-started>`_
        #. Add the Rust toolchain binaries to your system path environment variable (if installed at the default location, the path to the binaries will usually something like ``C:\Users\<user-name>\.cargo\bin``)

    .. tab-item:: Windows (MSYS2)

        #. Install the `MSYS2 toolchain <https://www.msys2.org/>`_.
        #. Add the MSYS2 toolchain to your `system path environment variable <https://www.eukhost.com/kb/how-to-add-to-the-path-on-windows-10-and-windows-11/>`_ (if installed at the default location, the path to the binaries will usually look something like ``C:\msys64\mingw64\bin``)
        #. Install Rust from the `Rust installation page <https://www.rust-lang.org/learn/get-started>`_, using the custom installation method and switching to the GNU version of Rust if desired. Alternatively, the default installation can be used and the GNU version can be set as the default in the last step.
        #. Add the Rust toolchain binaries to your system path environment variable (if installed at the default location, the path to the binaries will usually look something like ``C:\Users\<user-name>\.cargo\bin``)
        #. Now, run the following commands to switch to the GNU version of Rust if you did not already switch in Step 3:

        .. code-block:: shell

            rustup toolchain install stable-x86_64-pc-windows-gnu
            rustup default stable-x86_64-pc-windows-gnu
        
        .. note::

            You may need to restart your terminal before running the previous commands.

    .. tab-item:: WSL

        Install Rust from the `Rust installation page <https://www.rust-lang.org/learn/get-started>`_. At the time of writing, the command was

        .. code-block:: shell

            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    
    .. tab-item:: Linux

        Install Rust from the `Rust installation page <https://www.rust-lang.org/learn/get-started>`_

After these steps are completed correctly, the ``pip install`` command should execute successfully.
