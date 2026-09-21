from brownie import project, network, accounts

p = project.load(project_path=".", name="chainServerTest")
p.load_config()
from brownie.project.chainServerTest import NetworkManager, clientManager, watermarkNegotiation

network.connect('development')

deployer = accounts[0]
print("Deployer account:", deployer.address, "| Balance:", deployer.balance())

nm = NetworkManager.deploy({'from': deployer})
print("NetworkManager deployed at:", nm.address)

nm.fl_init(5, 10, "FedAvg", {'from': deployer})
print("FL round initialized:", nm.fl_info())

client1 = accounts[1]
nm.client_regist({'from': client1})
print("Client registered:", client1.address, "| is registered:", nm.regist_result(client1.address))

# Contract stores result = a * 10^b with both a and b unsigned, so accuracy
# 0.87 is encoded as an integer percentage instead: a=87, b=0 -> "87".
nm.upload_result(87, 0, 1, {'from': client1})
print("Uploaded result for client1:", nm.regist_result(client1.address))

network.disconnect()
print("SUCCESS: full deploy+interact cycle worked")
