/*
 *  Copyright 2014-2017 Hui Zhang
 *  Previous contribution by Nick Rutar 
 *  All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

package blame;

import java.awt.*;
import javax.swing.*;

public class VariableMetaData extends JFrame {
    //============================================== instance variables
   JTextArea _resultArea = new JTextArea(6, 20);
        
    //====================================================== constructor
    public VariableMetaData() {
        //... Set textarea's initial text, scrolling, and border.
        _resultArea.setText("Enter more text to see scrollbars");
        JScrollPane scrollingArea = new JScrollPane(_resultArea);
        
        //... Get the content pane, set layout, add to center
        JPanel content = new JPanel();
        content.setLayout(new BorderLayout());
        content.add(scrollingArea, BorderLayout.CENTER);
        
        //... Set window characteristics.
        this.setContentPane(content);
        this.setTitle("TextAreaDemo B");
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.pack();
    }
    
}

