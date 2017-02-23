# Load Qt framework
# adjust qtlibdir path and version info if necessary

@osx_only begin
    const qtlibdir = joinpath(homedir(),"Qt/5.3/clang_64/lib/")
    const QtCore = joinpath(qtlibdir,"QtCore.framework/")
    const QtWidgets = joinpath(qtlibdir,"QtWidgets.framework/")

    addHeaderDir(qtlibdir; isFramework = true, kind = C_System)

    dlopen(joinpath(QtCore,"QtCore_debug"))
    addHeaderDir(joinpath(QtCore,"Headers"), kind = C_System)
    addHeaderDir(joinpath(QtCore,"Headers/5.3.2/QtCore"))
    cxxinclude(joinpath(QtCore,"Headers/5.3.2/QtCore/private/qcoreapplication_p.h"))

    dlopen(joinpath(QtWidgets,"QtWidgets"))
    addHeaderDir(joinpath(QtWidgets,"Headers"), kind = C_System)
end

@linux_only begin
    const qtincdir = isdir("/usr/include/qt5") ? "/usr/include/qt5" : "/usr/include/x86_64-linux-gnu/qt5"
    const qtlibdir = "/usr/lib/x86_64-linux-gnu/"
    const QtWidgets = joinpath(qtincdir,"QtWidgets")

    addHeaderDir(qtincdir, kind = C_System)
    addHeaderDir(QtWidgets, kind = C_System)

    Libdl.dlopen(joinpath(qtlibdir,"libQt5Core.so"), Libdl.RTLD_GLOBAL)
    Libdl.dlopen(joinpath(qtlibdir,"libQt5Gui.so"), Libdl.RTLD_GLOBAL)
    Libdl.dlopen(joinpath(qtlibdir,"libQt5Widgets.so"), Libdl.RTLD_GLOBAL)
end

cxxinclude("QApplication", isAngled=true)
cxxinclude("QMessageBox", isAngled=true)
cxxinclude("QFileDialog", isAngled=true)

# convertQString
cxx"""
    const char* convertQString(QString myString) {
        std::string s = myString.toLocal8Bit().constData();
        // alternatives: toAscii(), toLatin1(), toUtf8()
        // std::cout << s << std::endl;
        const char* c = s.c_str();
        return(c);
    }
"""

# Open file with QFileDialog, returns Julia String
function getOpenFileName()
    const a = "FileOpen"
    argv = Ptr{Uint8}[pointer(a),C_NULL]  # const char** argv
    argc = [int32(1)]                     # int agrc
    app = @cxx QApplication(*(pointer(argc)),pointer(argv))
    Qname = @cxx QFileDialog::getOpenFileName(cast(C_NULL, pcpp"QWidget"),
       pointer("Open File..."), pointer(homedir()), pointer("Image Files (*.png *.jpg *.jpeg *.bmp *.tif *tiff)"));
    bytestring(@cxx convertQString(Qname));  # somewhow necessary to remove the first string
    filename = bytestring(@cxx convertQString(Qname))
end

# Open file with QFileDialog, returns Julia String
function getSaveFileName()
    const a = "FileSave"
    argv = Ptr{Uint8}[pointer(a),C_NULL]  # const char** argv
    argc = [int32(1)]                     # int agrc
    app = @cxx QApplication(*(pointer(argc)),pointer(argv))
    Qname = @cxx QFileDialog::getSaveFileName(cast(C_NULL, pcpp"QWidget"),
       pointer("Save File..."), pointer(homedir()), pointer("Image Files (*.png *.jpg *.jpeg *.bmp *.tif *tiff)"));
    bytestring(@cxx convertQString(Qname));  # somewhow necessary to remove the first string
    filename = bytestring(@cxx convertQString(Qname))
end
