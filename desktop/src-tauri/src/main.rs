// En release no se abre una consola junto a la ventana (Windows).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    perceptron_desktop_lib::run()
}
