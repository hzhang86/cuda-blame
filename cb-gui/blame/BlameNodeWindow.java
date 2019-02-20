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

import javax.swing.JFrame;

import java.awt.Component;
import java.awt.Container;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;


import javax.swing.JLabel;
import javax.swing.JList;

import javax.swing.JScrollPane;

import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

public class BlameNodeWindow extends JFrame {

	private Container c;
	private GridBagLayout gbl;
	private GridBagConstraints gbc;
	
	private JLabel nodeLabel, instanceLabel;
	public JList nodeList, instanceList;
	private JScrollPane nBody, iBody;
	
    private void addComponent( Component com, int row, int column,int width, int height)
    {
        // set gridx and gridy                                                        
    	gbc.gridx = column;
        gbc.gridy = row;

        gbc.gridwidth = height;
        gbc.gridheight = width;

        // set constraints                                                            
        gbl.setConstraints(com, gbc);
        c.add(com);
    }
	
	BlameNodeWindow(BlameContainer bc)
	{
		super ("BlameNodeWindow");
		c = getContentPane();
		gbl = new GridBagLayout();
		c.setLayout(gbl);
		
		gbc = new GridBagConstraints();
		

		
		// NodeList
		nodeList = new JList(bc.getNodes());
		nodeList.setFixedCellWidth(500);
		nodeList.setVisibleRowCount(10);
		nodeList.setSelectedIndex(0);
		nBody = new JScrollPane(nodeList);
		
		gbc.weightx = 1000;
		gbc.weighty = 20;
		gbc.fill = GridBagConstraints.BOTH;
		
		// 1 x 1 field at row 1, column 0
		addComponent(nBody, 1, 0, 1, 1);
		
		// Instance List (based on node)
		ParentData startBD = bc.getNodes().elementAt(0);
		instanceList = new JList(startBD.instances);
		
		if (startBD == null)
			System.err.println("StartBD is NULL!");
		
		if (startBD.instances != null && startBD.instances.size() > 0)
		{
			instanceList.setFixedCellWidth(500);
			instanceList.setVisibleRowCount(10);
		
			instanceList.setSelectedIndex(0);
		}
		iBody = new JScrollPane(instanceList);
		
		gbc.weightx = 1000;
		gbc.weighty = 20;
		gbc.fill = GridBagConstraints.BOTH;
		
		// 1 x 1 field at row 1, column 1
		addComponent(iBody, 1, 1, 1, 1);
		
		gbc.weightx = 0;
		gbc.weighty = 0;
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.NORTHWEST;
		
		
//		 1x1 field at row 0, column 0
		nodeLabel = new JLabel("Node Name");
		addComponent(nodeLabel, 0, 0, 1, 1);
	
		// 1x1 field at row 0, column 1
		instanceLabel = new JLabel("Instances");
		addComponent(instanceLabel, 0, 1, 1, 1);
		
		setSize( 250, 600);
		
		
		nodeList.addListSelectionListener(
				new ListSelectionListener() {
					public void valueChanged( ListSelectionEvent e)
					{
						if ( e.getValueIsAdjusting() == false)
						{
							// Get the current stack frame
							
							int index = nodeList.getSelectedIndex();
							ParentData bd = (ParentData)nodeList.getSelectedValue();
							
							
							if (index >= 0)
							{
								if (bd.instances != null && bd.instances.size() > 0)
								{
									instanceList.setListData(bd.instances);
									instanceList.setSelectedIndex(0);
								}
								
								
								iBody.revalidate();
								iBody.repaint();
							}
						}
					}
				}
		);
		
		
		
		
		
		
		
			
	}
}
